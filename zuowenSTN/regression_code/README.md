First stage code for localization network in the 2-stage spatial transformation invariance code.

Goal: try to write a general framework. 
    should consider further experiments on :
        1. different datasets
        2. different architecture
        3. different spatial transformation (currently only trans, rot)
        4. possible for adding CoRe as regularizer

Current Main TODOs:
1. modify resnet for regression task (mostly done)
2. modify previous train.py to train_reg.py, specifically:
            A. after reading dataset using cifarXX[X]_input.py, and applying spatial attack (chosen from our allowed set)
               replace the old y labels with rot, trans_x, trans_y
            B. modify all accuracy related code to abs_error, relative_error, histogram
      
3. try some experiments to check if the algo is running correctly. 
   MAKE SURE we are not using unrelated stuffs from the old framework!(since I directly building code upon the old one)
            C. [optional] check correlation between error&rotation angle, error&translation_pixel
            D. [optional] visualize the result for sanity check
4. experiment architecture X loss functions to find most suitable combination for this task