/* 
 * Author - Dustin Franklin (Nvidia Jetson Developer)
 * Modified by - Sahil Juneja, Kyle Stewart-Frantz
 *
 */

#include "ArmPlugin.h"
#include "PropPlugin.h"

#include "cudaMappedMemory.h"
#include "cudaPlanar.h"

#define PI 3.141592653589793238462643383279502884197169f

#define JOINT_MIN	-0.75f
#define JOINT_MAX	 2.0f

// Turn on velocity based control
#define VELOCITY_CONTROL false
#define VELOCITY_MIN -0.2f
#define VELOCITY_MAX  0.2f

// Define DQN API Settings

#define INPUT_CHANNELS 3
#define ALLOW_RANDOM true
#define DEBUG_DQN false
#define GAMMA 0.9f
#define EPS_START 0.9f
#define EPS_END 0.05f
#define EPS_DECAY 100

/*
/ TODO - Tune the following hyperparameters
/
*/

#define INPUT_WIDTH   64
#define INPUT_HEIGHT  64
#define NUM_ACTIONS DOF*2
#define OPTIMIZER "RMSprop"
#define LEARNING_RATE 0.005f
#define REPLAY_MEMORY 10000
#define BATCH_SIZE 16
#define USE_LSTM true
#define LSTM_SIZE 64

/*
/ TODO - Define Reward Parameters
/
*/

#define TIMEOUT_REWARD -1.0f
#define GROUND_COLLISION_REWARD -10.0f
#define ARM_COLLISION_REWARD 5.0f
#define GRIPPER_COLLISION_REWARD 50.0f
#define ALPHA 0.3f

// Define Object Names
#define WORLD_NAME "arm_world"
#define PROP_NAME  "tube"
#define GRIP_NAME  "gripper_middle"

// Define Collision Parameters
#define COLLISION_FILTER "ground_plane::link::collision"
#define COLLISION_ITEM   "tube::tube_link::tube_collision"
#define COLLISION_GRIPPER  "arm::gripperbase::gripper_link"
#define COLLISION_ARM "arm::link2::collision2"

// Animation Steps
#define ANIMATION_STEPS 1000

// Set Debug Mode
#define DEBUG false

// Lock base rotation DOF (Add dof in header file if off)
#define LOCKBASE true


namespace gazebo
{
 
// register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(ArmPlugin)


// constructor
ArmPlugin::ArmPlugin() : ModelPlugin(), cameraNode(new gazebo::transport::Node()), collisionNode(new gazebo::transport::Node())
{
	printf("ArmPlugin::ArmPlugin()\n");

	for( uint32_t n=0; n < DOF; n++ )
		resetPos[n] = 0.0f;

	resetPos[1] = 0.25;

	for( uint32_t n=0; n < DOF; n++ )
	{
		ref[n] = resetPos[n]; //JOINT_MIN;
		vel[n] = 0.0f;
	}

	agent 	       = NULL;
	inputState       = NULL;
	inputBuffer[0]   = NULL;
	inputBuffer[1]   = NULL;
	inputBufferSize  = 0;
	inputRawWidth    = 0;
	inputRawHeight   = 0;
	actionJointDelta = 0.15f;
	actionVelDelta   = 0.1f;
	maxEpisodeLength = 100;
	episodeFrames    = 0;

	newState         = false;
	newReward        = false;
	endEpisode       = false;
	gripperContact   = false;
	armContact       = false;
	timeout          = false;
	groundContact    = false;
	rewardHistory    = 0.0f;
	testAnimation    = true;
	loopAnimation    = false;
	animationStep    = 0;
	lastGoalDistance = 0.0f;
	avgGoalDelta     = 0.0f;
	successfulGrabs = 0;
  	failedGrabs     = 0;
	armContacts     = 0;
	gripperContacts = 0;
	totalRuns       = 0;
	timeouts        = 0;
	groundContacts  = 0;
  	
  	outfile.open("results.csv", std::ofstream::trunc);
  	outfile << "Run," << "Wins," << "Win Percent," << "Losses," << "Loss Percent," << "Arm Contacts,"
      		<< "Arm Contact Percent," << "Gripper Contacts," << "Gripper Contact Percent," << "Ground Contacts,"
      		<< "Ground Contact Percent," << "Timeouts," << "Timeout Percent," << std::endl;
  	outfile.close();
}


// Load
void ArmPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) 
{
	printf("ArmPlugin::Load('%s')\n", _parent->GetName().c_str());

	// Store the pointer to the model
	this->model = _parent;
	this->j2_controller = new physics::JointController(model);

	// Create our node for camera communication
	cameraNode->Init();
	
	/*
	/ Done - Subscribe to camera topic
	/
	*/
	
	cameraSub = cameraNode->Subscribe("/gazebo/arm_world/camera/link/camera/image", &ArmPlugin::onCameraMsg, this);

	// Create our node for collision detection
	collisionNode->Init();
		
	/*
	/ Done - Subscribe to prop collision topic
	/
	*/
	
	collisionSub = collisionNode->Subscribe("/gazebo/arm_world/tube/tube_link/my_contact", &ArmPlugin::onCollisionMsg, this);

	// Listen to the update event. This event is broadcast every simulation iteration.
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&ArmPlugin::OnUpdate, this, _1));
}


// CreateAgent
bool ArmPlugin::createAgent()
{
	if( agent != NULL )
		return true;

			
	/*
	/ Done - Create DQN Agent
	/
	*/
	
	agent = dqnAgent::Create(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, NUM_ACTIONS, 
					   OPTIMIZER, LEARNING_RATE, REPLAY_MEMORY,
					   BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY,
					   USE_LSTM, LSTM_SIZE, ALLOW_RANDOM, DEBUG_DQN);

	if( !agent )
	{
		printf("ArmPlugin - failed to create DQN agent\n");
		return false;
	}

	// Allocate the python tensor for passing the camera state
		
	inputState = Tensor::Alloc(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

	if( !inputState )
	{
		printf("ArmPlugin - failed to allocate %ux%ux%u Tensor\n", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
		return false;
	}

	return true;
}



// onCameraMsg
void ArmPlugin::onCameraMsg(ConstImageStampedPtr &_msg)
{
	// don't process the image if the agent hasn't been created yet
	if( !agent )
		return;

	// check the validity of the message contents
	if( !_msg )
	{
		printf("ArmPlugin - recieved NULL message\n");
		return;
	}

	// retrieve image dimensions
	
	const int width  = _msg->image().width();
	const int height = _msg->image().height();
	const int bpp    = (_msg->image().step() / _msg->image().width()) * 8;	// bits per pixel
	const int size   = _msg->image().data().size();

	if( bpp != 24 )
	{
		printf("ArmPlugin - expected 24BPP uchar3 image from camera, got %i\n", bpp);
		return;
	}

	// allocate temp image if necessary
	if( !inputBuffer[0] || size != inputBufferSize )
	{
		if( !cudaAllocMapped(&inputBuffer[0], &inputBuffer[1], size) )
		{
			printf("ArmPlugin - cudaAllocMapped() failed to allocate %i bytes\n", size);
			return;
		}

		printf("ArmPlugin - allocated camera img buffer %ix%i  %i bpp  %i bytes\n", width, height, bpp, size);
		
		inputBufferSize = size;
		inputRawWidth   = width;
		inputRawHeight  = height;
	}

	memcpy(inputBuffer[0], _msg->image().data().c_str(), inputBufferSize);
	newState = true;

	if(DEBUG){printf("camera %i x %i  %i bpp  %i bytes\n", width, height, bpp, size);}

}


// onCollisionMsg
void ArmPlugin::onCollisionMsg(ConstContactsPtr &contacts)
{
	//if(DEBUG){printf("collision callback (%u contacts)\n", contacts->contact_size());}

	if( testAnimation )
		return;

	for (unsigned int i = 0; i < contacts->contact_size(); ++i)
	{
		const char * object1 = contacts->contact(i).collision1().c_str();
		const char * object2 = contacts->contact(i).collision2().c_str();
		
		//if the contact is with the ground, don't mention it here
		if( strcmp(object1, COLLISION_FILTER)==0 || strcmp(object2, COLLISION_FILTER)==0 )
			continue;

		if(DEBUG){std::cout << "Collision between[" << object1
			     << "] and [" << object2 << "]\n";}
	
		/*
		/ DONE - Check if there is collision between the arm and object, then issue learning reward
		/
		*/
		
		if(strcmp(object1, COLLISION_ITEM)==0 || strcmp(object2, COLLISION_ITEM)==0)
		{
			if(strcmp(object1, COLLISION_ARM)==0 || strcmp(object2, COLLISION_ARM)==0)
			{
				printf("ARM CONTACT, EOE\n");				
				rewardHistory = ARM_COLLISION_REWARD;
				newReward  = true;
				endEpisode = true;
				armContact = true;
			}else if(strcmp(object1, COLLISION_GRIPPER)==0 || strcmp(object2, COLLISION_GRIPPER)==0){
				printf("GRIPPER CONTACT, EOE\n");
				rewardHistory = GRIPPER_COLLISION_REWARD;
				newReward  = true;
				endEpisode = true;
				gripperContact = true;
				armContact = true;
			}
		}
	}
}


// upon recieving a new frame, update the AI agent
bool ArmPlugin::updateAgent()
{
	// convert uchar3 input from camera to planar BGR
	if( CUDA_FAILED(cudaPackedToPlanarBGR((uchar3*)inputBuffer[1], inputRawWidth, inputRawHeight,
							         inputState->gpuPtr, INPUT_WIDTH, INPUT_HEIGHT)) )
	{
		printf("ArmPlugin - failed to convert %zux%zu image to %ux%u planar BGR image\n",
			   inputRawWidth, inputRawHeight, INPUT_WIDTH, INPUT_HEIGHT);

		return false;
	}

	// select the next action
	int action = 0;

	if( !agent->NextAction(inputState, &action) )
	{
		printf("ArmPlugin - failed to generate agent's next action\n");
		return false;
	}

	// make sure the selected action is in-bounds
	if( action < 0 || action >= DOF * 2 )
	{
		printf("ArmPlugin - agent selected invalid action, %i\n", action);
		return false;
	}

	if(DEBUG){printf("ArmPlugin - agent selected action %i\n", action);}



#if VELOCITY_CONTROL
	// if the action is even, increase the joint position by the delta parameter
	// if the action is odd,  decrease the joint position by the delta parameter

		
	/*
	/ Done - Increase or decrease the joint velocity based on whether the action is even or odd
	/
	*/
	
	float velocity = 0.0; // TODO - Set joint velocity based on whether action is even or odd.

	//if even, add vel delta to current velocity
	//otherwise, subtract vel delta from current velocity
	velocity = vel[action/2] + (action%2==0 ? actionVelDelta : -actionVelDelta);

	if( velocity < VELOCITY_MIN )
		velocity = VELOCITY_MIN;

	if( velocity > VELOCITY_MAX )
		velocity = VELOCITY_MAX;

	vel[action/2] = velocity;
	
	for( uint32_t n=0; n < DOF; n++ )
	{
		ref[n] += vel[n];

		if( ref[n] < JOINT_MIN )
		{
			ref[n] = JOINT_MIN;
			vel[n] = 0.0f;
		}
		else if( ref[n] > JOINT_MAX )
		{
			ref[n] = JOINT_MAX;
			vel[n] = 0.0f;
		}
	}
#else
	
	/*
	/ Done - Increase or decrease the joint position based on whether the action is even or odd
	/
	*/
	float joint = 0.0; // TODO - Set joint position based on whether action is even or odd.

	//if even, add joint delta to current position
	//otherwise, subtract joint delta from current position
	joint = ref[action/2] + (action%2 == 0 ? actionJointDelta : -actionJointDelta);

	// limit the joint to the specified range
	if( joint < JOINT_MIN )
		joint = JOINT_MIN;
	
	if( joint > JOINT_MAX )
		joint = JOINT_MAX;

	ref[action/2] = joint;

#endif

	return true;
}


// update joint reference positions, returns true if positions have been modified
bool ArmPlugin::updateJoints()
{
	if( testAnimation )	// test sequence
	{
		const float step = (JOINT_MAX - JOINT_MIN) * (float(1.0f) / float(ANIMATION_STEPS));

#if 0
		// range of motion
		if( animationStep < ANIMATION_STEPS )
		{
			animationStep++;
			printf("animation step %u\n", animationStep);

			for( uint32_t n=0; n < DOF; n++ )
				ref[n] = JOINT_MIN + step * float(animationStep);
		}
		else if( animationStep < ANIMATION_STEPS * 2 )
		{			
			animationStep++;
			printf("animation step %u\n", animationStep);

			for( uint32_t n=0; n < DOF; n++ )
				ref[n] = JOINT_MAX - step * float(animationStep-ANIMATION_STEPS);
		}
		else
		{
			animationStep = 0;

		}

#else
		// return to base position
		for( uint32_t n=0; n < DOF; n++ )
		{
			
			if( ref[n] < resetPos[n] )
				ref[n] += step;
			else if( ref[n] > resetPos[n] )
				ref[n] -= step;

			if( ref[n] < JOINT_MIN )
				ref[n] = JOINT_MIN;
			else if( ref[n] > JOINT_MAX )
				ref[n] = JOINT_MAX;
			
		}

		animationStep++;
#endif

		// reset and loop the animation
		if( animationStep > ANIMATION_STEPS )
		{
			animationStep = 0;
			
			if( !loopAnimation )
				testAnimation = false;
		}
		else if( animationStep == ANIMATION_STEPS / 2 )
		{	
			//ResetPropDynamics(); //original code to set object in same spot
			RandomizeProps(); //move object x position randomly
		}

		return true;
	}

	else if( newState && agent != NULL )
	{
		// update the AI agent when new camera frame is ready
		episodeFrames++;

		if(DEBUG){printf("episode frame = %i\n", episodeFrames);}

		// reset camera ready flag
		newState = false;

		if( updateAgent() )
			return true;
	}

	return false;
}


// get the servo center for a particular degree of freedom
float ArmPlugin::resetPosition( uint32_t dof )
{
	return resetPos[dof];
}


// compute the distance between two bounding boxes
static float BoxDistance(const math::Box& a, const math::Box& b)
{
	float sqrDist = 0;

	if( b.max.x < a.min.x )
	{
		float d = b.max.x - a.min.x;
		sqrDist += d * d;
	}
	else if( b.min.x > a.max.x )
	{
		float d = b.min.x - a.max.x;
		sqrDist += d * d;
	}

	if( b.max.y < a.min.y )
	{
		float d = b.max.y - a.min.y;
		sqrDist += d * d;
	}
	else if( b.min.y > a.max.y )
	{
		float d = b.min.y - a.max.y;
		sqrDist += d * d;
	}

	if( b.max.z < a.min.z )
	{
		float d = b.max.z - a.min.z;
		sqrDist += d * d;
	}
	else if( b.min.z > a.max.z )
	{
		float d = b.min.z - a.max.z;
		sqrDist += d * d;
	}
	
	return sqrtf(sqrDist);
}


// called by the world update start event
void ArmPlugin::OnUpdate(const common::UpdateInfo& updateInfo)
{
	// deferred loading of the agent (this is to prevent Gazebo black/frozen display)
	if( !agent && updateInfo.simTime.Float() > 1.5f )
	{
		if( !createAgent() )
			return;
	}

	// verify that the agent is loaded
	if( !agent )
		return;

	// determine if we have new camera state and need to update the agent
	const bool hadNewState = newState && !testAnimation;

	// update the robot positions with vision/DQN
	if( updateJoints() )
	{
		double angle(1);

#if LOCKBASE
		j2_controller->SetJointPosition(this->model->GetJoint("base"), 	0);
		j2_controller->SetJointPosition(this->model->GetJoint("joint1"),  ref[0]);
		j2_controller->SetJointPosition(this->model->GetJoint("joint2"),  ref[1]);

#else
		j2_controller->SetJointPosition(this->model->GetJoint("base"), 	 ref[0]); 
		j2_controller->SetJointPosition(this->model->GetJoint("joint1"),  ref[1]);
		j2_controller->SetJointPosition(this->model->GetJoint("joint2"),  ref[2]);
#endif
	}

	// episode timeout
	if( maxEpisodeLength > 0 && episodeFrames > maxEpisodeLength )
	{
		printf("ArmPlugin - triggering EOE, episode has exceeded %i frames\n", maxEpisodeLength);
		rewardHistory = TIMEOUT_REWARD;
		newReward     = true;
		endEpisode    = true;
	}

	// if an EOE reward hasn't already been issued, compute an intermediary reward
	if( hadNewState && !newReward )
	{
		PropPlugin* prop = GetPropByName(PROP_NAME);

		if( !prop )
		{
			printf("ArmPlugin - failed to find Prop '%s'\n", PROP_NAME);
			return;
		}

		// get the bounding box for the prop object
		const math::Box& propBBox = prop->model->GetBoundingBox();
		physics::LinkPtr gripper  = model->GetLink(GRIP_NAME);

		if( !gripper )
		{
			printf("ArmPlugin - failed to find Gripper '%s'\n", GRIP_NAME);
			return;
		}

		// get the bounding box for the gripper		
		const math::Box& gripBBox = gripper->GetBoundingBox();
		const float groundContact = 0.05f;
		
		/*
		/ Done - set appropriate Reward for robot hitting the ground.
		/
		*/
		
		//check if gripper hitting ground (min z position of bounding box within small buffer)
		bool checkGroundContact = gripBBox.min.z <= 0;//groundContact;		

		if(checkGroundContact)
		{
						
			printf("GROUND CONTACT, EOE\n");

			rewardHistory = GROUND_COLLISION_REWARD;
			newReward     = true;
			endEpisode    = true;
		}
		
		
		/*
		/ Done - Issue an interim reward based on the distance to the object
		/
		*/ 
		
	
		if(!checkGroundContact)
		{
			const float distGoal = BoxDistance(gripBBox, propBBox); // compute the reward from distance to the goal

			if(DEBUG){printf("distance('%s', '%s') = %f\n", gripper->GetName().c_str(), prop->model->GetName().c_str(), distGoal);}
			
			
			if( episodeFrames > 1 )
			{
				const float distDelta  = lastGoalDistance - distGoal;

				// compute the smoothed moving average of the delta of the distance to the goal
				avgGoalDelta  = avgGoalDelta*ALPHA + distDelta*(1-ALPHA);
				rewardHistory = avgGoalDelta;				
				newReward     = true;	
			}

			lastGoalDistance = distGoal;
		}
	}

	// issue rewards and train DQN
	if( newReward && agent != NULL )
	{
		if(DEBUG){printf("ArmPlugin - issuing reward %f, EOE=%s  %s\n", rewardHistory, endEpisode ? "true" : "false", (rewardHistory > 0.1f) ? "POS+" :(rewardHistory > 0.0f) ? "POS" : (rewardHistory < 0.0f) ? "    NEG" : "       ZERO");}
		agent->NextReward(rewardHistory, endEpisode);

		// reset reward indicator
		newReward = false;

		// reset for next episode
		if( endEpisode )
		{
			testAnimation    = true;	// reset the robot to base position
			loopAnimation    = false;
			endEpisode       = false;
			episodeFrames    = 0;
			lastGoalDistance = 0.0f;
			avgGoalDelta     = 0.0f;

			// track the number of wins and agent accuracy
			if( rewardHistory > 0.0 )
            {	
            	successfulGrabs++;
            } else {
            	failedGrabs++;
            }

			if( armContact )
			{
				armContact = false;				
				armContacts++;
			}

			if( gripperContact )
			{
				gripperContact = false;				
				gripperContacts++;
			}

			if( timeout )
			{
				timeout = false;
				timeouts++;
			}
			
			if( groundContact )
			{
				groundContact = false;
				groundContacts++;
			}	

			totalRuns++;

			outfile.open("results.csv", std::ios_base::app);
          	outfile << totalRuns - 1 << ","
              		<< successfulGrabs << ","
              		<< (float) successfulGrabs / (float) totalRuns << ","
              		<< failedGrabs << ","
              		<< (float) failedGrabs << ","
              		<< armContacts << ","
              		<< (float) armContacts / (float) totalRuns << ","
              		<< gripperContacts << ","
              		<< (float) gripperContacts / (float) totalRuns << ","
              		<< groundContacts << ","
              		<< (float) groundContacts / (float) totalRuns << ","
              		<< timeouts << ","
				    << (float) timeouts / (float) totalRuns << std::endl;

          	outfile.close();

			printf("Current Accuracy: %0.4f (%03u of %03u) Arm Contacts,\n"
			       "                  %0.4f (%03u of %03u) Gripper Contacts,\n"
			       "                  (reward=%+0.2f %s)\n",
				float(armContacts)/float(totalRuns), armContacts, totalRuns,
				float(gripperContacts)/float(totalRuns), gripperContacts, totalRuns,
				rewardHistory, (rewardHistory > 0.0 ? "WIN" : "LOSS"));


			for( uint32_t n=0; n < DOF; n++ )
				vel[n] = 0.0f;
		}
	}
}

}
