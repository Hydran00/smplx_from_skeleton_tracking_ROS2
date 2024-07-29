ZED_BODY_38= {
    "PELVIS" : 0,
    "SPINE_1" : 1,
    "SPINE_2" : 2,
    "SPINE_3" : 3,
    "NECK" : 4,
    "NOSE" : 5,
    "LEFT_EYE" : 6,
    "RIGHT_EYE" : 7,
    "LEFT_EAR" : 8,
    "RIGHT_EAR" : 9,
    "LEFT_CLAVICLE" : 10,
    "RIGHT_CLAVICLE" : 11,
    "LEFT_SHOULDER" : 12,
    "RIGHT_SHOULDER" : 13,
    "LEFT_ELBOW" : 14,
    "RIGHT_ELBOW" : 15,
    "LEFT_WRIST" : 16,
    "RIGHT_WRIST" : 17,
    "LEFT_HIP" : 18,
    "RIGHT_HIP" : 19,
    "LEFT_KNEE" : 20,
    "RIGHT_KNEE" : 21,
    "LEFT_ANKLE" : 22,
    "RIGHT_ANKLE" : 23,
    "LEFT_BIG_TOE" : 24,
    "RIGHT_BIG_TOE" : 25,
    "LEFT_SMALL_TOE" : 26,
    "RIGHT_SMALL_TOE" : 27,
    "LEFT_HEEL" : 28,
    "RIGHT_HEEL" : 29,
    # Hands
    "LEFT_HAND_THUMB_4" : 30, # tip
    "RIGHT_HAND_THUMB_4" : 31,
    "LEFT_HAND_INDEX_1" : 32, # knuckle
    "RIGHT_HAND_INDEX_1" : 33,
    "LEFT_HAND_MIDDLE_4" : 34, # tip
    "RIGHT_HAND_MIDDLE_4" : 35,
    "LEFT_HAND_PINKY_1" : 36, # knuckle
    "RIGHT_HAND_PINK_1" : 37,
    "LAST" : 38
}
SMPL_BODY_24 = [
    "pelvis", 
    "left_hip", # 0
    "right_hip", # 1
    "spine1", # 2
    "left_knee", # 3
    "right_knee", # 4
    "spine2", # 5
    "left_ankle", # 6
    "right_ankle", # 7
    "spine3", # 8
    "left_foot", # 9
    "right_foot", # 10
    "neck", # 11
    "left_collar", # 12
    "right_collar", # 13
    "head", # 14
    "left_shoulder", # 15
    "right_shoulder", # 16
    "left_elbow", # 17
    "right_elbow", # 18
    "left_wrist", # 19
    "right_wrist", # 20
    "left_hand", # 21
    "right_hand", # 22
]

ZED_BODY_38_TO_SMPL_BODY_24 = [
    ZED_BODY_38["PELVIS"], #0
    ZED_BODY_38["LEFT_HIP"], #0
    ZED_BODY_38["RIGHT_HIP"], #1
    ZED_BODY_38["SPINE_1"], #2
    ZED_BODY_38["LEFT_KNEE"], #3
    ZED_BODY_38["RIGHT_KNEE"], #4
    ZED_BODY_38["SPINE_2"], #5
    ZED_BODY_38["LEFT_ANKLE"], #6
    ZED_BODY_38["RIGHT_ANKLE"], #7
    ZED_BODY_38["SPINE_3"], #8 
    ZED_BODY_38["LEFT_BIG_TOE"], #9
    ZED_BODY_38["RIGHT_BIG_TOE"], #10
    ZED_BODY_38["NECK"], #11
    ZED_BODY_38["LEFT_CLAVICLE"], #12
    ZED_BODY_38["RIGHT_CLAVICLE"], #13
    ZED_BODY_38["NOSE"], #14
    ZED_BODY_38["LEFT_SHOULDER"], #15
    ZED_BODY_38["RIGHT_SHOULDER"], #16
    ZED_BODY_38["LEFT_ELBOW"], #17
    ZED_BODY_38["RIGHT_ELBOW"], #18
    ZED_BODY_38["LEFT_WRIST"], #19
    ZED_BODY_38["RIGHT_WRIST"], #20
    ZED_BODY_38["LEFT_HAND_THUMB_4"], #21
    ZED_BODY_38["RIGHT_HAND_THUMB_4"] #22
]

ZED_BODY_38_TO_SMPL_BODY_24_MIRROR = [
    ZED_BODY_38["PELVIS"], #0
    ZED_BODY_38["RIGHT_HIP"], #0
    ZED_BODY_38["LEFT_HIP"], #1
    ZED_BODY_38["SPINE_1"], #2
    ZED_BODY_38["RIGHT_KNEE"], #3
    ZED_BODY_38["LEFT_KNEE"], #4
    ZED_BODY_38["SPINE_2"], #5
    ZED_BODY_38["RIGHT_ANKLE"], #6
    ZED_BODY_38["LEFT_ANKLE"], #7
    ZED_BODY_38["SPINE_3"], #8
    ZED_BODY_38["RIGHT_BIG_TOE"], #9
    ZED_BODY_38["LEFT_BIG_TOE"], #10
    ZED_BODY_38["NECK"], #11
    ZED_BODY_38["RIGHT_CLAVICLE"], #12
    ZED_BODY_38["LEFT_CLAVICLE"], #13
    ZED_BODY_38["NOSE"], #14
    ZED_BODY_38["RIGHT_SHOULDER"], #15
    ZED_BODY_38["LEFT_SHOULDER"], #16
    ZED_BODY_38["RIGHT_ELBOW"], #17
    ZED_BODY_38["LEFT_ELBOW"], #18
    ZED_BODY_38["RIGHT_WRIST"], #19
    ZED_BODY_38["LEFT_WRIST"], #20
    ZED_BODY_38["RIGHT_HAND_THUMB_4"], #21
    ZED_BODY_38["LEFT_HAND_THUMB_4"] #22
]


SMPL_LANDMARK_INDICES = {"HEAD_TOP": 412,
                    "HEAD_LEFT_TEMPLE": 166,
                    "NECK_ADAM_APPLE": 3050,
                    "LEFT_HEEL": 3458,
                    "RIGHT_HEEL": 6858,
                    "LEFT_NIPPLE": 3042,
                    "RIGHT_NIPPLE": 6489,

                    "SHOULDER_TOP": 3068,
                    "INSEAM_POINT": 3149,
                    "BELLY_BUTTON": 3501,
                    "BACK_BELLY_BUTTON": 3022,
                    "CROTCH": 1210,
                    "PUBIC_BONE": 3145,
                    "RIGHT_WRIST": 5559,
                    "LEFT_WRIST": 2241,
                    "RIGHT_BICEP": 4855,
                    "RIGHT_FOREARM": 5197,
                    "LEFT_SHOULDER": 3011,
                    "RIGHT_SHOULDER": 6470,
                    "LOW_LEFT_HIP": 3134,
                    "LEFT_THIGH": 947,
                    "LEFT_CALF": 1103,
                    "LEFT_ANKLE": 3325
}
