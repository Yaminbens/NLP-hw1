
###Basic parameters###


#LEARN
#step size
LAMBDAb = 5
#maximum iteration
MAXITERb = 30
#accuracy for optimization
FACTRb = 10.0

#vector file save
VECSAVEb = "v_basic_30_10_L5_WT3"

#WORD PARSING
#word tag minimum occurancy
#WORDTAGFACTOR = 3


#EVALUATION
#vector to eval
VECTESTb = VECSAVEb



###Complex parameters###


#LEARN
#step size
LAMBDAc = 0.1
#maximum iteration
MAXITERc = 31
#accuracy for optimization
FACTRc = 10.0


#WORD PARSING
#word tag minimum occurancy
WORDTAGFACTOR = 3

#vector file save
VECSAVEc = "v_complex_"+str(MAXITERc)+"_"+str(FACTRc)+"_L"+str(LAMBDAc)+"_WT"+str(WORDTAGFACTOR)

#EVALUATION
#vector to eval
VECTESTc = VECSAVEc

