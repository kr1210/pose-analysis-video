import numpy as np
import logging
logger = logging.getLogger("pose_match")



def decide_torso_shoulders_incl(max_euclid_distance_torso, transformation_matrix, eucld_tresh, rotation_tresh,
                                max_euclid_distance_shoulders, shoulder_thresh):

        #Calcuation rotation of transformation
        rotation_1 = np.abs(np.math.atan2(-transformation_matrix[0][1], transformation_matrix[0][0]) * 57.3)
        rotation_2 = np.abs(np.math.atan2(transformation_matrix[1][0], transformation_matrix[1][1]) * 57.3)
        rot_max = max(rotation_2, rotation_1)

        print(" --- Evaluate Torso---")
        print(" max eucldis: {}  thresh({})".format(max_euclid_distance_torso, eucld_tresh))
        print(" max rot:     %s  thresh(%s)", rot_max, rotation_tresh)
        print(" max shoulder:%s  thresh(%s)", max_euclid_distance_shoulders, shoulder_thresh)

        torso_error = ((eucld_tresh - max_euclid_distance_torso)/eucld_tresh) * 100
        shoulder_error = ((shoulder_thresh - max_euclid_distance_shoulders)/shoulder_thresh) * 100    
        #print(int(torso_error), int(shoulder_error))
        upper_body_error = (torso_error + shoulder_error)/2
        upper_body_error_true = 0
        if (upper_body_error>90):
            upper_body_error_true = upper_body_error
        print(">>>>>>>>>>>>>TORSO-SCORE :", upper_body_error)
        if (max_euclid_distance_torso <= eucld_tresh and rot_max <= rotation_tresh):

        
            if (max_euclid_distance_shoulders <= shoulder_thresh):
                print("\t ->#TORSO MATCH#")
                return True,upper_body_error
            else:
                print("!!!!!TORSO NO MATCH Schouder error te groot!!!!")

        # Geen match
        print("\t ->#TORSO NO MATCH#")

        return False,upper_body_error_true


#Evaluate legs ..
def decide_legs(max_error, transformation_matrix, eucld_tresh, rotation_tresh):
    rotation_1 = np.abs(np.math.atan2(-transformation_matrix[0][1], transformation_matrix[0][0]) * 57.3)
    rotation_2 = np.abs(np.math.atan2(transformation_matrix[1][0], transformation_matrix[1][1]) * 57.3)
    rot_max = max(rotation_2, rotation_1)


    logger.debug(" --- Evaluate Legs---")
    logger.debug(" max eucldis: %s thresh(%s)", max_error, eucld_tresh)
    logger.debug(" max rot:     %s thresh(%s)", rot_max, rotation_tresh)

    legs_error = ((eucld_tresh - max_error)/eucld_tresh) * 100
    print(">>>>>>>>>>LEGS-SCORE: ", int(legs_error))
    if (max_error <= eucld_tresh and rot_max <= rotation_tresh):
        print("\t ->#LEGS MATCH#")
        return True

    
    print("\t ->#LEGS NO-MATCH#")
    return False


def max_euclidean_distance_shoulders(model_torso, input_transformed_torso):
    maxError_torso = np.abs(model_torso - input_transformed_torso)

    euclDis_torso = ((maxError_torso[:, 0]) ** 2 + maxError_torso[:, 1] ** 2) ** 0.5

    # Opgelet!! als nek er niet in zit is linker schouder = index 0 en rechterschouder = index 3
    # indien nek incl = > index 1 en index 4
    maxError_shoulder = max([euclDis_torso[1], euclDis_torso[4]])
    return maxError_shoulder

def max_euclidean_distance(model, transformed_input):

    manhattan_distance = np.abs(model - transformed_input)

    euclidean_distance = ((manhattan_distance[:, 0]) ** 2 + manhattan_distance[:, 1] ** 2) ** 0.5

    return max(euclidean_distance)
