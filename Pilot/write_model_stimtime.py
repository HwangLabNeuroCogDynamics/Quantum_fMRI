
import pandas as pd
import glob
import numpy as np
import os.path
import matplotlib.pyplot as plt
import numpy.matlib as mb

# function to write AFNI style stim timing file
def write_stimtime(filepath, inputvec):
	''' short hand function to write AFNI style stimtime'''
	#if os.path.isfile(filepath) is False:
	f = open(filepath, 'w')
	for val in inputvec[0]:
		if val =='*':
			f.write(val + '\n')
		else:
			# this is to dealt with some weird formating issue
			#f.write(np.array2string(val))
			#print np.array2string(np.around(val,4))
			f.write(np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','') + '\n') 
			#np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','') + '\n'
			#np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','')[2:-1] + '\n'
	f.close()


def extract_model_inputs(df):
  # create new column variable of "run number" 
  df['Run'] = df['block']
  cur_df = df
  tState = np.where(cur_df['state']==-1, 1, 0) # recode state (1 now 0 ... -1 now 1 ... this flip matches model better than just changing -1)
  tTask = np.zeros(len(tState)) # face=1, scene=0
  TaskPerformed = np.zeros(len(tState)) # face=1, scene=0

  cue_color = np.array(cur_df['cue']).flatten() # recode cue_color (now -1 (red) is 0 ... and 1 (blue) is 1 )
  cue_color = np.where(cue_color==-1, 0, cue_color) # recode cue_color (now -1 (red) is 0 ... and 1 (blue) is 1 )
  # now change amb to match tProp setup
  tProp = np.array(cur_df['amb_r']).flatten() # set as proportion of red

  tResp=[]
  target = list(cur_df['target'])
  correct = np.array(cur_df['correct']).flatten()
  subj_resp = np.array(cur_df['subj_resp']).flatten()
  for trl, trl_target in enumerate(target):
      if trl_target == "face":
          tTask[trl] = 1 # change task to 1
      if correct[trl] == 1:
          tResp.append(0)
      else:
          if subj_resp[trl]==-1:
              # no response... cound as wrong task for now
              tResp.append(2)
          elif trl_target == "face":
              # resp should be 0 or 1
              if subj_resp[trl] > 1:
                  # wrong task
                  tResp.append(2)
              else:
                  # right task but wrong answer
                  tResp.append(1)
          elif trl_target == "scene":
              # resp should be 2 or 3
              if subj_resp[trl] < 2:
                  # wrong task
                  tResp.append(2)
              else:
                  # right task but wrong answer
                  tResp.append(1)
      if subj_resp[trl] < 2:
          # they think they should do the scene task
          TaskPerformed[trl]=1
  tResp = np.array(tResp).flatten()

  print("\ntState:\n", tState)
  print("\ntTask:\n", tTask)
  print("\ntProp:\n", tProp)
  print("\ntResp:\n", tResp)

  return tState, tTask, tProp, tResp


def SequenceSimulation(nTrials, switchRange, thetas):

  #rng; # NEED TO FIND PYTHON EQUIVALENT

  tState = np.array([])
  cState = 0
  l = 0
  r = np.array(range(switchRange[0],(switchRange[1]+1),1)) 

  # set up state based on switch range
  while l < nTrials:
    #r = r[np.random.permutation(len(r))]
    #tState = np.concatenate( (tState, mb.repmat(cState, 1, r[0]).flatten()), axis=None )
    r = np.random.choice([15,25])
    tState = np.concatenate( (tState, mb.repmat(cState, 1, r).flatten()), axis=None )
    cState = 1 - cState
    l = l + r
    #l = l + r[0]

  tState = tState[:nTrials] # make sure tState is only as long as the number of trials
  
  tProp = np.random.rand(nTrials) # randomly generate proportion (cue)... btw 0 and 1
  possible_prop = [0.11111111,0.33333333,0.44444444,0.48,0.52,0.55555556,0.66666667,0.88888889]
  # avoid extreme values in simulation
  tProp[tProp < 0.05] = 0.05
  tProp[tProp > 0.95] = 0.95

  tResp = np.zeros((tState.shape))
  tTask = np.zeros((tState.shape))

  # loop through all trials
  for i in range(nTrials):
    np.random.shuffle(possible_prop)
    tProp[i] = possible_prop[0]

    x = np.exp(thetas[0]) * np.log( (tProp[i]/(1 - tProp[i])) )
    p = 1 / (1 + np.exp((-1*x)))

    # agent got confused within 3 trials after switch
    if (i > 4) and ( (tState[i] != tState[(i-1)]) or (tState[i] != tState[(i-2)]) or (tState[i] != tState[(i-3)])):
      # if we are within 3 trials of a switch then assume they're still doing the old task
      cState = 1 - tState[i]
    else:
      # if we are within the first 3 task trials assume they are randomly guessing
      if i < 4:
        if np.random.rand() < 0.5:
          cState = 0
        else:
          cState = 1
      else:
        cState = tState[i]

    if ( (tProp[i] > 0.5) and (tState[i] == 0) ) or ( (tProp[i] < 0.5) and (tState[i] > 0) ):
      tTask[i] = 0
    else:
      tTask[i] = 1

    # rTask is the simulated task the agent thinks it should do
    if np.random.rand() < p:
      #color 0
      if cState == 0:
        rTask = 0
      else:
        rTask = 1
    else:
      #color 1
      if cState == 0:
        rTask = 1
      else:
        rTask = 0

    if tTask[i] != rTask:
      tResp[i] = 2
    else:
      if rTask == 0:
        if np.random.rand() < thetas[1]:
          tResp[i] = 1
        else:
          tResp[i] = 0
      else:
        if np.random.rand() < thetas[2]:
          tResp[i] = 1
        else:
          tResp[i] = 0

  return tState, tProp, tResp, tTask


# this is the model, I modified the JD to export trial by trail posterior distribution
def MPEModel(tState, tProp, tResp):
    
    sE = []
    tE = []

    #start with uniform prior
    #pRange = cell(1, 4);
    pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    #range of logistic function slope parameter
    #pRange{1} = -5:0.05:5;
    #range of face task error rate
    #pRange{2} = 0:0.01:0.2;
    #range of scene task error rate
    #pRange{3} = 0:0.01:0.2;
    #diffusion across trials
    #pRange{4} = 0:0.01:0.5;
    #dim = [2, length(pRange{1}), length(pRange{2}), length(pRange{3}), length(pRange{4})];
    dim = np.array( [2, len(pRange['lfsp']), len(pRange['fter']), len(pRange['ster']), len(pRange['diff_at'])] )
    
    total = 1
    # for i = 1:length(dim)
    # total = total * dim(i);
    # end
    # jd = ones(dim) / total;
    for c_dim in dim:
        total = total * c_dim
    jd = np.ones(dim) / total   # the division of total is to normalize values into probability
    
    # to store trial by trial jd
    all_jd = np.ones(np.array([len(tState), 2, len(pRange['lfsp'])]))

    #likelihood of getting correct response
    #ll = zeros(size(jd));
    ll = np.zeros(jd.shape)

    #tProp = log(tProp ./ (1 - tProp));

    ###############????????????????????????????????????????????????????######
    # Question 3, here why taking the log of ratio of dots for the two colors?
    # Make it more normal?
    ###############????????????????????????????????????????????????????######
    tProp = np.log(tProp / (1 - tProp))  

    # for i = 1 : length(tState)
    # if mod(i, 20) == 0
    #     disp([num2str(i) ' trials have been simulated.']);
    # end
    for it in range(len(tState)):
        if (it % 50) == 0:
            print(str(it), ' trials have been simulated.')
        #diffusion of joint distribution over trials
        # for i4 = 1 : dim(4)
        #     x = (jd(1, :, :, :, i4) + jd(2, :, :, :, i4)) / 2;

        #     jd(1, :, :, :, i4) = jd(1, :, :, :, i4) * (1 - pRange{4}(i4)) + pRange{4}(i4) * x;
        #     jd(2, :, :, :, i4) = jd(2, :, :, :, i4) * (1 - pRange{4}(i4)) + pRange{4}(i4) * x;
        # end

        ###############????????????????????????????????????????????????????######    
        ###############????????????????????????????????????????????????????######
        # Question 4, what is the purpose of this block of code below? 
        # it appears the effect is essentially to smooth paramters between task state 0 and 1??
        
        for i4 in range(dim[4]):
            x = (jd[0, :, :, :, i4] + jd[1, :, :, :, i4]) / 2

            jd[0, :, :, :, i4] = jd[0, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
            jd[1, :, :, :, i4] = jd[1, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
        #add estimate as marginalized distribution
        #sE(end + 1) = sum(sum(sum(sum(jd(1, :, :, :, :)))));
        sE.append(np.sum(jd[0, :, :, :, :]))
        ###############????????????????????????????????????????????????????######
        ###############????????????????????????????????????????????????????######


        #first color logit greater than 0 means it is dominant, less than 0
        #means other color dominant

        ############################################
        #### below is the mapping between color ratio, task state, and task
        # tState = 0, tProp<0 : tTask = 1
        # tState = 0, tProp>0 : tTask = 0 
        # tState = 1, tProp<0 : tTask = 0
        # tState = 1, tProp<0 : tTask = 1
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0
        else:
            tTask = 1
        ################################################


        # S_Theta1 = squeeze(sum(sum(sum(jd, 3), 4), 5));
        # tE(end + 1) = 0;
        # here marginalize theta1
        S_Theta1 = np.squeeze(np.sum(np.sum(np.sum(jd, 4), 3), 2)) # double check that axes are correct
        
        tE.append(0)
        for i0 in range(dim[0]):
            for i1 in range(dim[1]):

                ###############????????????????????????????????????????????????????######
                ###############????????????????????????????????????????????????????######
                # Question 5:
                # here this block to logit transform  color ratio to probability of color decision.
                # I believe the formula from JF's white borad was:
                # p(pColor | tProp) = 1 / (1 + exp( -1* theta1 * log(tProp) ))
                #  
                # and convert the color decision probability to task belief (0 to 1)
                # The last line task multiply with S_Theta1 (the marginalized set belief and logistic function slope),
                # is it to weight the task belief by the probability of each logistic slope param in the distribution for each task set...?

                                                     # Quesion 6 here   
                theta1 = np.exp(pRange['lfsp'][i1])  # why take the exponent of logisitc function slope??
                pColor = 1 / (1 + np.exp((-1*theta1) * tProp[it]))
                if i0 == 0:
                    pTask0 = pColor
                else:
                    pTask0 = 1 - pColor

                #print(tE[-1] + pTask0 * S_Theta1[i0, i1])
                tE[-1] = tE[-1] + pTask0 * S_Theta1[i0, i1] 
                ###############????????????????????????????????????????????????????######
                ###############????????????????????????????????????????????????????######


                for i2 in range(dim[2]):
                    for i3 in range(dim[3]):
                        #pP is the likelihood, change this if there is only one type of error
                        # this is for separated response error and task error
                        
                        ###############????????????????????????????????????????????????????######
                        ###############????????????????????????????????????????????????????######
                        # question 7
                        # dont quite get what the liklihood here means given the three typoes of response below
                        # trial-wise response: 0 = correct, 1 = correct task wrong answer, 2 = wrong task
                        if tTask == 0:
                            pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2], (1 - pTask0)]
                        else:
                            pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3], pTask0]
                            #print(pP)
                        
                        #posterior, jd now is prior
                        ######
                        # question 8
                        # Here it appears you are using one of the three liklihood to update all joint thetas, will have to ask
                        # JF the logic of othis operation
                        ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[int(tResp[it])]
                        ###############????????????????????????????????????????????????????######
                        ###############????????????????????????????????????????????????????######

                        # #this is for only one type of error
                        # if tTask == 0:
                        #     pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2] + (1 - pTask0)]
                        # else:
                        #     pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3] + pTask0]
                        
                        # #posterior, jd now is prior
                        # if tResp(i) == 0:
                        #     ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[0]
                        # else:
                        #     ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[1]
                        
        #normalize
        jd = ll / np.sum(ll)
        all_jd[it,:,:] = np.sum(jd, axis=(2,3,4))

        mDist={}
        mDist['lfsp'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 2), 0))    # 4 3 2 0
        mDist['fter'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 1), 0))    # 4 3 1 0
        mDist['ster'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 2), 1), 0))    # 4 2 1 0
        mDist['diff_at'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 3), 2), 1), 0)) # 3 2 1 0
    
    return all_jd, sE, tE, mDist, pRange


################################################
###### create noise regressors, I know Evan wrote a function but I'm too lazy to figure out how to use it
################################################
confounds = sorted(glob.glob('/data/backed_up/shared/Quantum_data/MRI_data/fmriprep/fmriprep/sub-JHtest/func/*confounds_regressors.tsv'))
cdf = pd.read_csv(confounds[0], sep='\t')

for c in confounds[1:]:
	cdf =cdf.append(pd.read_csv(c, sep='\t'))

noise_mat = cdf.loc[:,['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 'trans_x', 'trans_x_derivative1',
       'trans_x_derivative1_power2', 'trans_x_power2', 'trans_y',
       'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
       'trans_z', 'trans_z_derivative1', 'trans_z_power2',
       'trans_z_derivative1_power2', 'rot_x', 'rot_x_derivative1',
       'rot_x_derivative1_power2', 'rot_x_power2', 'rot_y',
       'rot_y_derivative1', 'rot_y_power2', 'rot_y_derivative1_power2',
       'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2',
       'rot_z_power2']].values
noise_mat[np.isnan(noise_mat)]=0
np.savetxt('/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/noise.1D', noise_mat)


################################################
###### read psychopy model, run it through JF's model to reate model based regressors.... 
################################################
data_dir = "/mnt/cifs/rdss/rdss_kahwang/Quantum_Task/"
DM_list = glob.glob(os.path.join(data_dir, "Data_MRI", "sub-99997*Quantum*.csv"))

timing_logs = sorted(DM_list) #important to sort since glob seems to randomly order files.
num_runs = len(timing_logs)
#to check the order is correct, in ascending order
print('check timing logs are in ascending order:')
print(timing_logs) 

#load timing logs, concat depend on number of sessions
df = pd.read_csv(timing_logs[0])
for i in np.arange(1, num_runs, 1):
    df = df.append(pd.read_csv(timing_logs[i]))

tState, tTask, tProp, tResp = extract_model_inputs(df)

## fit model
all_jd, sE, tE, mDist, pRange = MPEModel(tState, tProp, tResp)

#all_jd is now the joint dist of theta0 and 1. We can use it to calculate the entropy of this distribution
# still don't quite understand how theta0 works, so sum?
t_theta1 = np.sum(all_jd, axis=1) # this is now the trial by trial theta1 distribution

from scipy.stats import entropy #this looks right https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
t_entropy = [] #entorpy of dist
t_entropy_change = [0.0] #signed change of entropy from previous trial
for t in range(len(tE)):
	t_entropy.append(entropy(t_theta1[t,:]))
	if t>0:
		t_entropy_change.append(entropy(t_theta1[t,:])-entropy(t_theta1[t-1,:]))
t_entropy = np.array(t_entropy)
t_entropy_change = np.array(t_entropy_change)


## plot
plt.plot(1-np.array(sE), 'blue', t_entropy_change, 'orange', tState, 'black')
plt.plot(np.array(tE), 'green', t_entropy_change, 'orange',)

plt.plot(np.array(sE), 'blue', tState, 'black',)

pRange_keys = ['lfsp', 'fter', 'ster', 'diff_at']
temp_SE = 1-np.array(sE)
plt.plot(tState, 'blue', temp_SE, 'orange')
plt.show()

plt.plot(tTask, 'blue', tE, 'orange')
plt.show()


############################################################
### now create regressors for these model params.
###############################################################
#convert state and task estimate into confidence, closer to 0.5 is high confence
state_est = abs(np.array(sE)-0.5) 
task_est = abs(np.array(tE)-0.5)

## create amplitude modulation record for 3dDeconvolve. See: https://afni.nimh.nih.gov/pub/dist/doc/misc/Decon/AMregression.pdf
Cue_state_stimtime = [['*']*num_runs]
Probe_state_stimtime = [['*']*num_runs] 
Cue_task_stimtime = [['*']*num_runs]
Probe_task_stimtime = [['*']*num_runs] 
Cue_entropy_stimtime = [['*']*num_runs]
Probe_entropy_stimtime = [['*']*num_runs] 
Cue_entropy_change_stimtime = [['*']*num_runs]
Probe_entropy_change_stimtime = [['*']*num_runs] 

j=0
for i, run in enumerate(np.arange(1,num_runs+1)): 
    run_df = df[df['Run']==run].reset_index() #"view" of block we are curreint sorting through
    Cue_state = [] 
    Probe_state = []
    Cue_task = [] 
    Probe_task = []
    Cue_entropy = [] 
    Probe_entropy = []
    Cue_entropy_change = [] 
    Probe_entropy_change = []

    for tr in np.arange(0, len(run_df)):  #this is to loop through trials
        Cue_state.append(state_est[j]) 		
        Probe_state.append(state_est[j]) 
        Cue_task.append(task_est[j]) 		
        Probe_task.append(task_est[j]) 
        Cue_entropy.append(t_entropy[j]) 		
        Probe_entropy.append(t_entropy[j]) 
        Cue_entropy_change.append(t_entropy_change[j]) 		
        Probe_entropy_change.append(t_entropy_change[j]) 
        j = j+1
        
    Cue_state_stimtime[0][i] = Cue_state
    Probe_state_stimtime[0][i] = Probe_state			
    Cue_task_stimtime[0][i] = Cue_task
    Probe_task_stimtime[0][i] = Probe_task	
    Cue_entropy_stimtime[0][i] = Cue_entropy
    Probe_entropy_stimtime[0][i] = Probe_entropy	
    Cue_entropy_change_stimtime[0][i] = Cue_entropy_change
    Probe_entropy_change_stimtime[0][i] = Probe_entropy_change	

#write out stimtime array to text file. 	
fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_state_am.1D' 
write_stimtime(fn, Cue_state_stimtime)		

fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_state_am.1D' 
write_stimtime(fn, Probe_state_stimtime)	

fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_task_am.1D' 
write_stimtime(fn, Cue_task_stimtime)	

fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_task_am.1D' 
write_stimtime(fn, Probe_task_stimtime)	

fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_am.1D' 
write_stimtime(fn, Cue_entropy_stimtime)		

fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_entropy_am.1D' 
write_stimtime(fn, Probe_entropy_stimtime)	

fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_entropy_change_am.1D' 
write_stimtime(fn, Cue_entropy_change_stimtime)	

fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_entropy_change_am.1D' 
write_stimtime(fn, Probe_entropy_change_stimtime)	


##### create cue and probe timing from psychopy output
#create FIR timing for cue and probe, extract trial timing
Cue_stimtime = [['*']*num_runs] #stimtime format, * for runs with no events of this stimulus class
Probe_stimtime = [['*']*num_runs] #create one of this for every condition

for i, run in enumerate(np.arange(1,num_runs+1)):  #loop through 12 runs
    run_df = df[df['Run']==run].reset_index() #"view" of block we are curreint sorting through
    Cue_times = [] #empty vector to store trial time info for the current block
    Probe_times = []

    for tr in np.arange(0, len(run_df)):  #this is to loop through trials
        Cue_times.append(run_df.loc[tr,'Time_Since_Run_Cue_Prez']) 		
        Probe_times.append(run_df.loc[tr,'Time_Since_Run_Photo_Prez']) 
        
    Cue_stimtime[0][i] = Cue_times	#put trial timing of each block into the stimtime array	
    Probe_stimtime[0][i] = Probe_times			


#write out stimtime array to text file. 	
fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/cue_stimtime.1D' 
write_stimtime(fn, Cue_stimtime)		
    
fn =  '/data/backed_up/shared/Quantum_data/MRI_data/3dDeconvolve/JHtest/probe_stimtime.1D' 
write_stimtime(fn, Probe_stimtime)		


#######################################################
##### simulate sequences
#######################################################

thetas = [np.log(1.2), 0.09, 0.07]
simtState, simtProp, simtResp, simtTask = SequenceSimulation(300, [15, 25], thetas)	
sim_all_jd, sim_sE, sim_tE, sim_mDist, sim_pRange = MPEModel(simtState, simtProp, simtResp)
sim_t_theta1 = np.sum(sim_all_jd, axis=1) # this is now the trial by trial theta1 distribution

from scipy.stats import entropy #this looks right https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
sim_t_entropy = [] #entorpy of dist
sim_t_entropy_change = [0.0] #signed change of entropy from previous trial
for t in range(len(sim_tE)):
	sim_t_entropy.append(entropy(sim_t_theta1[t,:]))
	if t>0:
		sim_t_entropy_change.append(entropy(sim_t_theta1[t,:])-entropy(sim_t_theta1[t-1,:]))
sim_t_entropy = np.array(sim_t_entropy)
sim_t_entropy_change = np.array(sim_t_entropy_change)


## plot
plt.plot(1-np.array(sim_sE), 'blue', sim_t_entropy_change, 'orange', simtState, 'black')


#######################################################
##### fit model and check entropy params from other behavioral pilots
#######################################################
data_dir = "/mnt/cifs/rdss/rdss_kahwang/Cued_Attention_Task/"
#DM_df = pd.read_csv("behav_pilot.csv")
#sdf = DM_df.loc[DM_df["Participant_ID"]==DM_df.Participant_ID.unique()[0]]
#s_tState, s_tTask, s_tProp, s_tResp = extract_model_inputs(sdf)

s_tState = np.load("sub-10218_model_inputs/sub-10218_tState.npy")
s_tTask = np.load("sub-10218_model_inputs/sub-10218_tTask.npy")
s_tProp = np.load("sub-10218_model_inputs/sub-10218_tProp.npy")
s_tResp = np.load("sub-10218_model_inputs/sub-10218_tResp.npy")
s_all_jd, s_sE, s_tE, s_mDist, s_pRange = MPEModel(s_tState, s_tProp, s_tResp)

s_t_theta1 = np.sum(sim_all_jd, axis=1) # this is now the trial by trial theta1 distribution
s_t_entropy = [] #entorpy of dist
s_t_entropy_change = [0.0] #signed change of entropy from previous trial
for t in range(len(s_tE)):
	s_t_entropy.append(entropy(s_t_theta1[t,:]))
	if t>0:
		s_t_entropy_change.append(entropy(s_t_theta1[t,:])-entropy(s_t_theta1[t-1,:]))
s_t_entropy = np.array(s_t_entropy)
s_t_entropy_change = np.array(s_t_entropy_change)
plt.plot(1-np.array(s_sE), 'blue', s_t_entropy_change, 'orange', s_tState, 'black')


#end