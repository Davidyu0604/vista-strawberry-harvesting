"""Experimental visual stalk centering; robot calibration/proprioception only."""
import mujoco
import numpy as np
from scipy.optimize import brentq
from grasp_pose import JawCalibration, solve_stalk_pose, fixed_roll_ik, RobotSelfCollision


class TrackingError(ValueError):
    """Measured motion protection stop, distinct from an invalid tool request."""


class PullPlanError(ValueError):
    """All bounded pull plans failed before any pull command was executed."""


class CenteredGrasp:
    def __init__(self, env, tick):
        self.e=env;self.tick=tick;self.cal=JawCalibration(env)
        self.scratch=mujoco.MjData(env.model)
        self.contact_opening=brentq(lambda q:self.cal.aperture(q)-.003,-.15,.2)
        self.point=None;self.axis=None;self.direction=None;self.pre_q=None
        self.carry_roll=None;self.observation_roll=None;self.blocked=0
        if env.meta.get('initial_arm_pose')=='search_stow':
            self.observation_roll=float(env.last_action[4])
            env.meta['observation_roll_rad']=self.observation_roll
        self.approach_commands=[];self.record_approach=False
        self.last_tracking=None
        self.self_collision=RobotSelfCollision(env)
        self.standoff=None

    def fk(self,q):
        d=self.scratch;d.qpos[:]=self.e.data.qpos;d.qpos[self.e.qa]=q
        mujoco.mj_kinematics(self.e.model,d)
        return d.site_xpos[self.e.tcp].copy()

    def command(self,goal,start=None):
        goal=np.asarray(goal,dtype=float);old=self.e.last_action.copy() if start is None else np.asarray(start).copy()
        if goal.shape!=(6,) or not np.isfinite(goal).all():raise ValueError('Invalid joint command')
        delta=goal-old;scale=min(1.,.04/max(np.max(np.abs(delta)),1e-10))
        origin=self.fk(old)
        while np.linalg.norm(self.fk(old+scale*delta)-origin)>.005:scale*=.8
        q=old+scale*delta
        return q,self.fk(q)

    def check_tracking(self,expected):
        error=float(np.linalg.norm(self.e.data.site_xpos[self.e.tcp]-expected))
        self.blocked=self.blocked+1 if error>.008 else 0
        self.last_tracking={'error_m':error,'consecutive_ticks':self.blocked,
                            'expected_tcp_m':np.asarray(expected).tolist(),
                            'actual_tcp_m':self.e.data.site_xpos[self.e.tcp].tolist()}
        if self.blocked>=3:
            exc=TrackingError(f'Tracking error above 8 mm for three ticks (actual {error*1000:.2f} mm)')
            exc.tracking=dict(self.last_tracking)
            raise exc
        return error

    def waypoint_ticks(self, goal, minimum):
        """Budget the existing rate-limited command path before starting motion."""
        q = self.e.last_action.copy()
        for steps in range(181):
            if np.max(np.abs(np.asarray(goal)-q)) < 1e-8:
                # Preserve the established duration when it already fits.
                return {'command_ticks': steps, 'minimum_ticks': minimum,
                        'total_ticks': minimum if steps < minimum else steps+10,
                        'command_tick_limit': 180}
            if steps == 180:
                break
            q, _ = self.command(goal, start=q)
        raise ValueError('Delivery command path exceeds 180 bounded ticks')

    def motion_tick(self,q,phase):
        if self.record_approach:
            old=self.e.last_action.copy()
            if not self.approach_commands or not np.array_equal(old,self.approach_commands[-1]):
                self.approach_commands.append(old)
        self.tick(q,phase)

    def goto(self,q,phase):
        for _ in range(180):
            if np.max(np.abs(q-self.e.last_action))<1e-8:return
            self.motion_tick(q,phase)
        raise ValueError('Centered motion timeout')

    def settle(self,q,phase,n=55):
        for _ in range(n):self.motion_tick(q,phase)

    def plan(self,point,axis):
        point=np.array(point,dtype=float,copy=True);axis=np.array(axis,dtype=float,copy=True)
        q,error=solve_stalk_pose(self.e,point,axis,self.e.state(),.2,self.cal.center(.2))
        direction=-self.e.ikdata.xmat[self.e.grip_body].reshape(3,3)[:,2].copy()
        direction[2]=0;direction/=max(np.linalg.norm(direction),1e-8)
        trials=[]
        for standoff in (.055,.045,.035):
            pre=point-standoff*direction
            pre_q,pre_error=solve_stalk_pose(self.e,pre,axis,self.e.state(),.2,self.cal.center(.2),direction)
            end_q,end_error=solve_stalk_pose(self.e,point,axis,pre_q,.2,self.cal.center(.2),direction)
            ik_error=max(error,pre_error,end_error)
            hit=self.self_collision.path_collision(self.e.last_action,pre_q) if ik_error<=.006 else None
            if hit is None and ik_error<=.006:hit=self.self_collision.collision(end_q)
            trials.append({'standoff_m':standoff,'max_ik_error_m':ik_error,'robot_self_collision':hit})
            if ik_error<=.006 and hit is None:
                return {'point':point,'axis':axis,'direction':direction,'pre_q':pre_q,
                        'max_ik_error_m':ik_error,'standoff_m':standoff,'preflight_trials':trials}
        exc=ValueError('No reachable self-collision-free centered pregrasp in bounded standoff candidates')
        exc.approach_trace={'preflight_trials':trials,'motion_started':False}
        raise exc

    def approach(self,stage,point,axis):
        self.approach_commands=[];self.record_approach=True
        try:return self._approach(stage,point,axis)
        finally:self.record_approach=False

    def _approach(self,stage,point,axis):
        e=self.e
        if stage=='pregrasp':
            plan=self.plan(point,axis)
            if plan['max_ik_error_m']>.006:raise ValueError('Centered pregrasp/endpoint unreachable')
            self.point=plan['point'];self.axis=plan['axis'];self.direction=plan['direction'];self.pre_q=plan['pre_q']
            self.standoff=plan['standoff_m'];self.preflight_trials=plan['preflight_trials']
            if self.observation_roll is None:self.observation_roll=float(e.last_action[4])
            self.carry_roll=None
            e.meta['observation_roll_rad']=self.observation_roll
            self.goto(self.pre_q,'visual_pregrasp');self.settle(self.pre_q,'visual_pregrasp')
        elif stage=='grasp':
            if self.point is None or not np.allclose(point,self.point,atol=1e-6,rtol=0):
                raise ValueError('Centered approach target changed; repeat pregrasp')
            origin=e.data.xpos[e.grip_body]+e.data.xmat[e.grip_body].reshape(3,3)@self.cal.center(.2)
            q=e.last_action.copy()
            for f in np.linspace(0,1,29)[1:]:
                q,error=solve_stalk_pose(e,origin+f*(self.point-origin),self.axis,q,.2,self.cal.center(.2),self.direction)
                if error>.006:raise ValueError('Centered approach waypoint unreachable')
                if self.self_collision.path_collision(e.last_action,q):raise ValueError('Centered grasp path would self-collide')
                self.goto(q,'visual_grasp')
            self.settle(q,'visual_grasp')
        else:raise ValueError('Unknown centered approach stage')
        desired=self.point-(self.standoff*self.direction if stage=='pregrasp' else 0)
        actual=e.data.xpos[e.grip_body]+e.data.xmat[e.grip_body].reshape(3,3)@self.cal.center(.2)
        error=float(np.linalg.norm(actual-desired))
        if error>.008:raise ValueError('Jaw center did not reach requested position')
        return {'jaw_center_error_m':error,'reference':'calibrated jaw center',
                'pregrasp_standoff_m':self.standoff,'preflight_trials':self.preflight_trials,
                'approach_direction_world':self.direction.tolist(),'approach_opening_rad':.2}

    def retreat(self):
        if not self.approach_commands:raise ValueError('No traversed centered approach to retreat')
        # Reverse traversed commands, including pregrasp motion. The pregrasp
        # destination itself is NOT a retreat when reaching it was what failed.
        self.record_approach=False;self.blocked=0
        for q in reversed(self.approach_commands):self.goto(q,'approach_retreat')
        self.settle(self.approach_commands[0],'approach_retreat')
        q=self.e.last_action.copy();q[5]=1.
        self.goto(q,'approach_retreat_open');self.settle(q,'approach_retreat_open',10)
        self.point=None;self.carry_roll=None
        self.approach_commands=[]

    def close(self):
        if self.point is None:raise ValueError('No confirmed centered approach')
        e=self.e
        for _ in range(60):
            start=e.last_action.copy();opening=max(-.15,start[5]-.01)
            center=self.cal.center(max(self.contact_opening,opening))
            q,error=solve_stalk_pose(e,self.point,self.axis,start,opening,center,self.direction)
            if error>.006:raise ValueError('Centered closing waypoint unreachable')
            if np.max(np.abs(q-start))>.04+1e-9 or np.linalg.norm(self.fk(q)-self.fk(start))>.005:
                raise ValueError('Centered closing exceeds motion limits')
            self.tick(q,'gripper')
        self.carry_roll=float(e.last_action[4])

    def ik(self,point,opening,pitch=0.):
        if self.carry_roll is None:return self.e.ik(point,opening,approach_pitch=pitch)
        return fixed_roll_ik(self.e,point,opening,self.carry_roll,pitch)

    def observation_plan(self, anchor, index, body_point=None):
        """Retract/raise; optionally orient the wrist camera at a measured body.

        Camera framing uses calibration and the visual surface only. It does
        not certify visibility, stalk identity, maturity or permission to grasp.
        """
        e=self.e
        if index not in (0,1) or e.state()[5]<.9:
            raise ValueError('Observation adjustment requires open gripper and remaining posture budget')
        offset=np.array([-.04 if index==0 else -.06,0.,.04])
        end=np.asarray(anchor['tcp_world_m'])+offset
        if body_point is None:
            self.observation_plan_diagnostics={'mode':'retract_raise'}
            return self._observation_path(end,anchor['roll'])
        body=np.asarray(body_point,dtype=float)
        if body.shape!=(3,) or not np.isfinite(body).all():raise ValueError('Finite measured body required')
        camera=e.model.camera('wrist').id
        # Include the body and its upper context, rather than aiming at a
        # guessed grasp point. Roll trials never move the base or close jaws.
        points=body+np.array([[0,0,0],[0,0,.04],[0,0,.075]])
        trials=[];choices=[]
        low,high=e.model.actuator_ctrlrange[4]
        for degrees in (0,-30,30,-60,60,-90,90):
            roll=float(anchor['roll']+np.deg2rad(degrees))
            if not low+.005<=roll<=high-.005:continue
            try:
                plan=self._observation_path(end,roll)
            except ValueError as exc:
                trials.append({'roll_offset_deg':degrees,'rejected':str(exc)})
                continue
            d=self.scratch;d.qpos[:]=e.data.qpos;d.qpos[e.qa]=plan[1][-1]
            mujoco.mj_kinematics(e.model,d);mujoco.mj_camlight(e.model,d)
            local=(points-d.cam_xpos[camera])@d.cam_xmat[camera].reshape(3,3)
            if np.any(local[:,2]>=-.01):
                trials.append({'roll_offset_deg':degrees,'rejected':'Measured body context behind camera'})
                continue
            scale=-local[:,2]*np.tan(np.deg2rad(e.model.cam_fovy[camera])/2)
            normalized=local[:,:2]/np.column_stack([scale*480/360,scale])
            edge=float(np.max(abs(normalized)))
            trials.append({'roll_offset_deg':degrees,'max_normalized_image_radius':edge,'ticks':len(plan[1])+10})
            if edge<.98:choices.append((edge+.001*abs(degrees),degrees,plan))
        self.observation_plan_diagnostics={'mode':'measured_body_camera_framing','trials':trials,'visibility_certified':False}
        if not choices:
            raise ValueError('No bounded observation posture frames the measured body and upper context')
        _,degrees,plan=min(choices,key=lambda x:x[0])
        self.observation_plan_diagnostics={'mode':'measured_body_camera_framing','selected_roll_offset_deg':degrees,
                                          'trials':trials,'visibility_certified':False}
        return plan

    def _observation_path(self,end,roll):
        e=self.e
        origin=e.data.site_xpos[e.tcp].copy();previous=e.last_action.copy();goals=[]
        start_roll=float(previous[4])
        for f in np.linspace(0,1,9)[1:]:
            q,error=fixed_roll_ik(e,origin+f*(end-origin),1.,start_roll+f*(roll-start_roll),0.,start=previous)
            if error>.006 or self.self_collision.path_collision(previous,q):
                raise ValueError('Observation posture path unreachable or robot self-collision')
            goals.append(q);previous=q
        commands=[];previous=e.last_action.copy()
        for goal in goals:
            while np.max(abs(goal-previous))>=1e-8:
                if len(commands)>=110:raise ValueError('Observation adjustment exceeds six-second budget')
                q,_=self.command(goal,start=previous)
                if np.max(abs(q-previous))<1e-12:raise ValueError('Observation adjustment cannot advance')
                commands.append(q);previous=q
        return end,commands

    def adjust_observation(self, plan):
        end,commands=plan
        base=self.e.base_pose().copy()
        for q,phase in [(q,'observation_adjust') for q in commands]+[(commands[-1] if commands else self.e.last_action,'observation_adjust_hold')]*10:
            self.tick(q,phase)
            if np.linalg.norm(self.e.base_pose()[:2]-base[:2])>.005:
                raise ValueError('Base moved during observation adjustment')
            if any(w.number for w in self.e.data.warning):
                raise ValueError('Physics warning during observation adjustment')
        error=float(np.linalg.norm(self.e.data.site_xpos[self.e.tcp]-end))
        if error>.008:raise ValueError('Observation posture endpoint did not settle')
        return {'tcp_error_m':error,'ticks':len(commands)+10,
                'observation_plan':getattr(self,'observation_plan_diagnostics',None),
                'note':'Observation posture only. Reobserve and refit the same candidate; no grasp authorized.'}

    def pull(self,delta,pitch,seconds,contact_check):
        """Withdraw briefly, then blend the remaining translation and bending.

        Preflight uses robot geometry only. Completion is not detachment proof.
        """
        e=self.e;delta=np.asarray(delta,dtype=float)
        if self.carry_roll is None:raise ValueError('Centered pull requires completed centered close')
        if delta.shape!=(3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>.04:
            raise ValueError('Centered pull displacement must be finite and at most 40 mm')
        if not np.isfinite(pitch) or abs(pitch)>np.pi/4 or seconds!=3:
            raise ValueError('Centered staged pull requires seconds=3 and pitch within 45 degrees')
        requested=delta.copy()
        alternatives=[('requested',delta)]
        # A world-X request can be nearly sideways at an oblique grasp. Try
        # retreating along the actual gripper approach, at the same distance,
        # height and pitch, only when the original path cannot be planned.
        outward=np.array(e.data.xmat[e.grip_body].reshape(3,3)[:,2],dtype=float,copy=True);outward[2]=0.
        if (abs(delta[2])<1e-10 and np.linalg.norm(outward)>1e-8
                and np.dot(outward,delta)>0):
            outward*=np.linalg.norm(delta)/np.linalg.norm(outward)
            if not np.allclose(outward,delta,atol=1e-6,rtol=0):
                alternatives.append(('gripper_outward',outward))
        trials=[]
        for path,delta in alternatives:
            try:
                commands,planned,budget,initial_pitch,end=self._plan_pull(delta,pitch,seconds)
            except ValueError as exc:
                trials.append({'path':path,'displacement_m':delta.tolist(),'rejected':str(exc)})
            else:
                trials.append({'path':path,'displacement_m':delta.tolist(),'planned_ticks':budget})
                break
        else:
            exc=PullPlanError('No feasible bounded pull path; no pull motion executed. '
                             + '; '.join(t['rejected'] for t in trials)
                             + '. Reinspect the grasp and revise the pull request.')
            exc.planning_trials=trials
            raise exc
        ticks=0;lost=0
        def tick(q,phase):
            nonlocal ticks,lost
            self.tick(q,phase);ticks+=1
            lost=0 if contact_check() else lost+1
            if lost>=4:raise ValueError('Staged pull stopped: external jaw contact lost for 0.2s')
            if any(w.number for w in e.data.warning):raise RuntimeError('Physics warning during staged pull')
        for q,phase in planned:tick(q,phase)
        while ticks<budget:tick(commands[-1][0],'bend_pull_hold')
        error=float(np.linalg.norm(e.data.site_xpos[e.tcp]-end))
        if error>.008:raise ValueError('Staged pull endpoint did not settle')
        return {'mode':'withdraw_then_blend','initial_pitch_rad':initial_pitch,'final_pitch_rad':pitch,
                'initial_withdraw_fraction':.4,'selected_path':path,'planning_trials':trials,
                'requested_displacement_m':requested.tolist(),
                'displacement_m':delta.tolist(),'ticks':ticks,'tcp_error_m':error,
                'requested_min_seconds':seconds,'actual_seconds':ticks*e.dt,'max_seconds':6,
                'note':'Motion completed only. Visual separation confirmation still required before delivery.'}

    def _plan_pull(self,delta,pitch,seconds):
        """Plan the entire pull from the current pose without advancing physics."""
        e=self.e;start=e.data.site_xpos[e.tcp].copy();end=start+delta
        initial_pitch=float(np.arcsin(np.clip(-e.data.xmat[e.grip_body].reshape(3,3)[2,2],-1,1)))
        # A full straight pull can load the still-attached stalk before any
        # bending starts. Preserve the endpoint, but begin bending after 40%
        # withdrawal and spread the motion over time instead of a fast burst.
        points=[(start+.4*f*delta,initial_pitch,'bend_pull_translate') for f in np.linspace(0,1,17)[1:]]
        points += [(start+(.4+.6*f)*delta,initial_pitch+f*(pitch-initial_pitch),'bend_pull_blend')
                   for f in np.linspace(0,1,49)[1:]]
        previous=e.last_action.copy();commands=[]
        for point,angle,phase in points:
            q,error=fixed_roll_ik(e,point,-.15,self.carry_roll,angle,start=previous)
            if error>.006:raise ValueError('Staged pull waypoint unreachable')
            if self.self_collision.path_collision(previous,q):raise ValueError('Staged pull path would self-collide')
            commands.append((q,phase));previous=q
        # Predict the bounded command duration before moving, rather than abort
        # halfway just because a different grasp pose needs more joint travel.
        planned=[];previous=e.last_action.copy();max_ticks=round(6/e.dt)
        for goal,phase in commands:
            while np.max(abs(goal-previous))>=1e-8:
                if len(planned)>=max_ticks-10:raise ValueError('Staged pull path exceeds six-second motion budget')
                q,_=self.command(goal,start=previous)
                if np.max(abs(q-previous))<1e-12:raise ValueError('Staged pull plan cannot advance')
                planned.append((q,phase));previous=q
        return commands,planned,max(round(seconds/e.dt),len(planned)+10),initial_pitch,end

    def restore_observation(self):
        from mounted_basket import ready_point
        if self.observation_roll is None:raise ValueError('No recorded observation wrist angle')
        if self.e.state()[5]<.8:raise ValueError('Restore observation only after opening and exiting basket')
        point=ready_point(self.e)
        q,error=fixed_roll_ik(self.e,point,1.,self.observation_roll)
        if error>.006:raise ValueError('Observation pose unreachable')
        self.goto(q,'delivery_restore_observation');self.settle(q,'delivery_restore_observation')
        position_error=float(np.linalg.norm(self.e.data.site_xpos[self.e.tcp]-point))
        roll_error=abs(float(self.e.state()[4])-self.observation_roll)
        if position_error>.008 or roll_error>.015:raise ValueError('Observation pose did not settle')
        self.carry_roll=None
        return {'restored':True,'tcp_error_m':position_error,'roll_error_rad':roll_error,
                'observation_roll_rad':self.observation_roll}
