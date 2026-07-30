---
title: "A Poor Man's Recipe to Robotic Machine Learning"
date: 2026-08-01
permalink: /poor-mans-recipe-to-robotic-machine-learning
math: true
tags:
  - robotics
  - reinforcement learning
---


Since we are all about advantages (and disadvantages) of LLMs, we quickly overlook how difficult it is for machines to assist us outside the virtual chat interfaces. People are getting older, work and professions are changing, but we still need to fold our clothes and sort our dishes at the end of the day, for a lot of days to come. The feeling lingers that the physical breakthrough for AI is just around the corner, ready to literally knock our front door down and aggressively scrub our toilets. 

| Gif |
| ---: |
| [ ![A robot stealing the show with its dance moves during a theatrical performance in China.](res/2025-10-01/dancing_robot.gif) ](res/2025-10-01/dancing_robot.gif) |
| A robot stealing the show with its dance moves during a theatrical performance in China. |

Still, we only see them dance around or controlled remotely to resemble us having a good, but not quite a productive time. We rather want to see them assist in hospitals and laboratories, for tedious tasks we (soon) do not find enough people to assign to. The problem is, the job needs training, and somehow we do not know how to train them appropriately yet. Humans would join a course or power through slides and online videos to find any recipes or instructions a training kickstart. We feel good by recognizing progress and hope of eventually achieving our goals. And for that, we can practice, step-by-step. Failure only affirms our persistence for success.

| Diagram |
| ---: |
| [ ![bike -> fall -> bike,bike -> fall -> bike,bike,bike,...](res/2025-10-01/bike_fall.png) ](res/2025-10-01/bike_fall.png) |
| How we learn to ride a bike: The more we fall, the better we stay on the bike afterwards. |

Greatly inspired by this heroic pattern, we cram the machine mind into an isolated simulation just for them to despair before our demands. Unlike a toddler in a play kitchen, a robot throws millions of attempts for the smallest tasks (even in toddler scale), like moving an object into a shaped hole.

| Image |
| ---: |
| [ ![A young girl in imaginative role-play using a wooden play kitchen.](res/2025-10-01/playing_girl.png) ](res/2025-10-01/playing_girl.png) |
| A young girl in imaginative role-play using a wooden play kitchen. |

And we are still in a befitted simulation. Turns out, humans are yet to be great teachers for machines that dream of autonomy, and not of procedural-tabular behaviour programs. And we already gifted them neural networks to come up with their own model and ideas to correlate the features of our shared world. (At the risk of conspiring hallucinations.)

| Diagram |
| ---: |
| [ ![Neural network](res/2025-10-01/neural_net.png) ](res/2025-10-01/neural_net.png) |
| How a small neural network might look like: Two arbitrary inputs (left) get "woven" through a net of four adjustable weights (middle) and spit out again as one (right). Modern AI. |

In this tutorial, we improve exactly on that: To be better teachers to our robots with the tools affordable to us today, and without the need to raise our own data center in the backyard. First, the mentioned Neural Networks to internalize what are teaching them. Second, a training loop by Reinforcement Learning to proceed how we are teaching them. And third, a way to present or inject exactly what task is asked and how to preferably go at it. The focus is especially on the latter part: We like to imagine machines that take a look at how we do things, and then learn on their own. We start just simple: Repeat simple movements in simulated 3D space.

| Note |
| ---: |
| **Our Ingredients**: <br> - Personal Computer <br/> - Neural Networks<br/> - Reinforcement Learning |

To show that, we setup a robotics machine learning environment on our computer with [Farama Gymnasium](https://robotics.farama.org/envs/fetch/push). The task is to control a simulated robot arm to move (“manipulate”) a box to a target position, also called a “Fetch-and-Push”.

| Gif |
| ---: |
| [ ![LOREM IPSUM](res/2025-10-01/fetchpush_success.gif) ](res/2025-10-01/fetchpush_success.gif) |
| LOREM IPSUM. |

| Diagram |
| ---: |
| [ ![LOREM IPSUM](res/2025-10-01/rl_interaction.png) ](res/2025-10-01/rl_interaction.png) |
| LOREM IPSUM |

The agent with its own Neural Network attempts to solve the task by forming (and exploring) its decisions on how to move the arm based on its interaction with the environment (while receiving an environmental state and reward). The state consists on kinematic data, and as such, it describes the position and velocity of the simulated objects relative to each other.

| Diagram |
| ---: |
| [ ![LOREM IPSUM](res/2025-10-01/robot_arm_space.png) ](res/2025-10-01/robot_arm_space.png) |
| space: actors & objs. (disps. + vels.) |

We would intuitively like the reward for the agent to be positive if an action closes the distance from the [end-effector](https://en.wikipedia.org/wiki/Robot_end_effector) (EE) to the box *or* the distance from the box to the target. But then, the agent (and honestly, humans too) would eventually choose to abuse the former distance: Moving towards the box, but never the box towards the target, for an “infinite money glitch”. The intended and subsequent action sequence to push the box in the correct direction is not obvious at all, since that might increase the first distance unintentionally — it is not worth the risk and difficulty.

<table>
  <thead>
    <tr>
      <th style="text-align: right;">Formula</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: right;">
$$
r_{naive}(s) =
\begin{cases}
1, & \text{if } v_{p,b} \lt 0 ,\\
1, & \text{if } v_{b,t} \lt 0 ,\\
-1, & \text{else.}
\end{cases}
$$ 
      </td>
    </tr>
    <tr>
      <td style="text-align: right;">LOREM IPSUM.</td>
    </tr>
  </tbody>
</table>

| Gif of push box and stop fail |



Even if a big reward is waiting at the goal, we would need to painfully wait for the agent to risk and discover the difficult follow-up by sheer chance. (This introduces the aspect of [*sample-efficiency*](https://ai.stackexchange.com/questions/5246/what-is-sample-efficiency-and-how-can-importance-sampling-be-used-to-achieve-it) in RL and, to an extent, [*sparse rewarding*](https://medium.com/@m.k.daaboul/dealing-with-sparse-reward-environments-38c0489c844d).) So, in defining the task or goal naively of closing two separate subdistances, we fail to communicate our intentions to the robot agent. This training “deadlock” is found in many composed task descriptions and prevents us from teaching machines ordinary manipulation in a straightforward manner: By telling them exactly how an action evaluates correctly at every step — a *clean* dense reward, so to speak.

## (Part 1) From Trajectory to Plan
<h3 style="text-align: right;"><i> > "A goal without a plan is just a wish."</i></h3>

In the simulation we are given the exact current state as a vector of coordinates and velocities for every object of interest. Suppose we (as an expert) have solved the task already and can demonstrate it. If we want to keep the universal notion of goal distance (and inherently a notion of progress), then we want to “record” the demonstration or demo in a composed and ordered way. We could assess the demo as a link of consecutive desired states, then we could follow from one state to the next and eventually end up at the final goal state with zero goal distance or 100% goal progress.

| \[start start to goal state\] |

We could use a neural network for that.

| current state \-\> demo recorder \-\> desired next state |

Two problems: First, every state transition is seen isolated, so the network might choose to “remember” a linear transition somewhere at the beginning in more detail rather than a non-linear transition that is crucial to the overall task — linearity is simply easier to detect and regress.

| Gif of isolated behaviour |

We need to add that not only single transitions are to be mapped as good as possible, but every transition must eventually lead to the goal state.

| NN with “foresight”/horizon (recorder inside monitor) |

Now the demo-recorder “understands” our intention of preserving a consistent chain of states toward the goal state: We just need to re-input the next state into the recorder multiple times to obtain a recorded trajectory or plan(\!) and a goal-distance by the sum of distances between the future states, starting from *any* possible state. A state can now be better than another if it is projected to lead to the goal state “faster” by its plan (horizon). A component that evaluates every state to its respective goal distance or progress extends a recorder/planner and we might call it a *progress* *monitor*. There lies the second problem: The monitor by its neural network needs to be trained thoroughly as well, and that independently from the RL-agent. By design, it outputs a next state or every possible input state, even if it has never been trained on that input before. This sounds both good and bad, depending on how well the training samples are distributed. In practice, if the monitor is confronted with an input that is well out-of-distribution, it will output rubbish.

| Gif of out-of-distr. fail |

To counter this during training, every input-output-pair is complemented by noisy inputs to the same output: We artificially cover a greater range of inputs around the original input.

| noisy inputs to demo recorder NN |

For every single input, we can therefore map multiple states more to the same next state, depending on the granularity and the distribution of the random noise generator. We can even cover more input states if we step into the (noisy) interval space between the input and the output:

| noisy space (colorized) |

Then the risk of hitting a major out-of-training-distribution input case is effectively decreased and the monitor is taught to evaluate a greater range of possible states. This directly improves the consistency of our goal distance.

| gif of consistent robot arm |

We have shown to reliably find a better goal distance that is rooted in demonstrations in 3D space. But we have not shown how to integrate the goal distance and its monitors into the RL loop — and whether just naively reward its decrease is enough. (It is not.) We also assumed perfect knowledge of the coordinates. How do we fare with less precise data, for example from a single RGB camera? We are going to tackle those challenges in the next parts of this tutorial series on robotic machine learning.

## (Part 2) From Plan to Reward
<h3 style="text-align: right;"><i> > "It’s all about the journey."</i></h3>

Since we have covered how to retrieve a goal distance from demonstrations as recorded trajectories around a neural network (Part 1), we now come to integrate successful learning from it as a clean reward signal. The reward is a crucial aspect in Reinforcement Learning. It dictates what actions of a RL agent are to be enforced throughout the states of the environment.

| Photo of Dog Training |

Naturally, if the goal distance changes, either by making or losing progress in the task, we want to change the choice of our actions accordingly, too. Let us assume now that the (human) demonstrations covered all possible scenarios how the task could be solved, and the monitor has learned all there is to *judge* it. (But not necessarily able to solve it by itself.) Their judgement of progress and a follow-up conversion into a clean and effective reward signal makes them a good *trainer*.

| (Evo of the trainer) |

How does a reward may look like? It could a be signal that outputs one of two differently signed numbers to affirm or reject any action possible (= dense reward), or it could be no reward at all for any action that reaches states outside of the goal state (= sparse reward).  
I am a proponent for the former signal as long as it is clean of misdirections — we want a trainer to ideally give frequent and *correct* feedback at all times during training. A sparse feedback signal suffers greatly with difficult tasks as it relies on the agent to discover (different) solutions on its own.  
With our clean goal distance available for all relevant states, we can opt to choose a dense signal that reward every effort to decrease the distance to the goal state and that punishes the other direction.

| reward func based on goal dist. |

A problem arises when the agent hits a state where the action to decrease the distance is not obvious, for example only a one single action among many others would lead to the goal. Since the reward signal does not differentiate between a good state-action pair that is closer to the goal and one that is more distant, the agent learns to “oscillate”: It moves back and forth to farm many rewards without attempting the next difficult section — it is just not worth the work

| gif: oscillation robot arm |

We can solve this by making the rewarder entity or component aware of progression: For every training episode, we only reward the agent for choosing actions that close the distance further than before in the same episode.

| reward formula with progress |

Additionally, we could restrict the reward even more to actions that close the distance enough to reach the goal in the remaining time (as we are able to count the steps until episode truncation).

| reward formula: directed, progressed, timed |

Now, there is no incentive for the agent to idle between states or to delay the goal approach.

## (Part 3) From Image to Trajectory
<h3 style="text-align: right;"><i> > "The real voyage of discovery consists not in seeking new landscapes, but in having new eyes."</i></h3>


Until now, we crucially assumed that the point coordinates of all relevant actors and objects of interest are delivered to us in perfect precision. In real applications, that is certainly almost never the case (and can be even detrimental in form of an overly strict trainer).  Realistically, we rely on optical sensors as in an ordinary camera to deliver us two-dimensional RGB projections of a three-dimensional world. Humans use a pair of eyes and experience to reconstruct coordinates and distances in the inner mind. To most extent, they can also survive on a single eye-sensor only.  
Machines can also be taught to estimate distances with sufficient precision to manipulate the world. We present an approach for ordinary 2D images of a typical RGB camera.

| graph: image to coords. |

Therefore, we need to map 2D image features onto 3D world coordinates. A Convoluted neural Network (CNN) does this for us — we can call it an “*extractor*” of 3D coordinates:

| extractor-component |

Similar to the goal-distance-monitor, the coordinate-extractor needs to be trained separately: A good trainer must be able to assess the state accurately, and for that, it needs to be trained themselves.

| evo of the trainer (aka. “video-to-reward”) |

The training must consist of as many different input-output configurations as possible to cover many projections of a diverse world whose “lost” dimension is hinted in the other two by shadows, reflections or other inferencing features that are not always obvious to us. Optimally, the training pairs cover most if not all possible coordinates the application can achieve.  
Having said that, it still would not be quite enough to just use the images as they are, and we apply some pre-processing tricks to assist the mapping training of the extractor:

1. We stack three consecutive frames of a video to present a sequence over time. The extractor can then correlate their image change to not only positional coordinates, but also velocities. And velocities are crucial to the progress monitor. (Why?)

| stacked input |

2. We provide for each 3D displacement an own image that centers the pixel of the displaced object and another image that centers the pixel of the origin (object) it is displaced from. For example, to estimate the coordinates position of the end-effector relative to the box, we pass *two* images: One of the end-effector tracked to the exact middle of that image, and one of the box tracked to the exact middle of that other image.

| stacked and tracked input |

We repeat this for every (relative) position/velocity we are interested in. Now this outsources some computing off our current components, and I believe the tasks of tracking sequence and pixel are (already) solved on their own.

| gif: fetch push in 64x64 rgb |

## (Part 4) From Reward to Policy
<h3 style="text-align: right;"><i> > "We are our decisions."</i></h3>

Now given a state estimation and a coherent reward, we can run the RL loop to train our robot agent to find the correct actions in a dynamic environment. In the end, this gives us an agent policy: For every state, there is an action that seemingly maximized the expected culmination of rewards over what time is left in the episode. There we can see why the reward design is so important — we want it to be as goal-oriented as possible, and we hope this to be the case with a reward signal that is (visually) trained on (human) expert demonstrations.  
Another key design is the choice of what state features the agent is able to “observe”. Since the agent policy is also a Neural Network, it takes an input vector of chosen state variables as well. An obvious choice would be to re-use or forward some subset from the trainer — what the teacher *needs* to recognize the progress, the actions of the student do, too.

| shared state graph |

In my Fetch-Push-experiments, it was enough to only forward the coordinates from the RGB extractor.  
However, due to the limitations of an optical sensor, the extracted coordinates won’t always be reliable. That eventually also stains our reward signal: Even when a state measures wrong enough only once in a while, it conflicts with the other states before or after, and the communicated progress/reward won’t be consistent along an otherwise correct trajectory. Without a correct feedback, the agent can get stuck very similar to situations where a task gets more difficult, for example when an unstable or unhandy object is grabbed or pushed closely — and the robot can’t find a way how.

| gif of non-blackout training |

In my experiments, I used again two tricks to help the training:

1. I noticed that the problem was more apparent the more features a task contained. Imagine learning a new sport, swimming for a good example. We first get taught the stroke in shallow water, or even outside the pool. Then we move into deeper water, practice our breathing, until we finally put everything together. Jumping without focus into the ocean instead would force us to apply everything at once right way, and even under perfect instructions, we still might fail miserably or give up because we are simply *overwhelmed*. Similar to a kickboard that helps the student *and* the teacher to focus on the legs during a swim, the rewarder component can zero-out some features randomly to focus only on the remaining dimensions. The selection is ideally made for a whole episode to keep the progress-/reward-signal consistent. We might call this a *feature blackout*. (Not to be confused with a feature *dropout*, where it changes every step.)

| photo: kickboard swimming |

| graph: feature blackout  |

2. After all that, I still noticed how the agent might still opt to “cash-in” easy rewards by oscillating or slowing down instead of attempting a difficult segment. There, a better trainer could look at the preceding actions and require a smooth execution of the trajectory towards the goal. Only if the last actions preserve the correct direction in a somewhat monotone manner, the following actions are projected to keep the goal *momentum*. Rewards are given (only) for goal-directed smoothness and monotonicity of  action-sequences. Then, not every smallest action in the right direction is rewarded in isolation, but only in nice interplay with its predecessors. Because the goal-distance or progress is scalar combination of different (dependent) dimensions, the overall smoothness *“bleeds*” into every one of those dimensions and the robot motion looks smoother as well.

| SCM/SAM-reward function |

Putting everything together, we end up with the following reward function:

| final reward function |

Coupled with a potent agent architecture and a suitable learning algorithm off-the-shelf (I used [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)), we not only taught a robot to solve a composite task by RL, but also taught a trainer to critic and control the training effectively for the actors and objects it has seen from demonstrations before. The trainer components are exchangeable so that different robot arms could still use the remaining pipeline or be used for other tasks.

However, we still want to find out how to select and track the points-of-interests on images automatically based on a task description. Before, I have done this manually. We also want to test the coordinate extractor with drastically changed perspectives, and therefore check the limitations of 2D images or the need of depth sensors. We also want to interconnect our training approach with language models to describe the task via natural language. And our robot to access the real world outside of its simulation. As you can see, we still have work to do for the next parts of this robotic machine learning series. **Stay tuned!**

| From Text to Task From Task to Image (From Image to Sim?) From Sim to Real |

| Next Trainer Evo |


| References |
| --- |
| Girl Image: Shlomaster @ Pixabay |