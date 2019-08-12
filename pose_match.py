import collections
import normalising
import prepocessing
import affine_transformation
import pose_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import logging
import numpy as np
import proc_do_it
import draw_humans
import cv2
import random

import matplotlib._png as png



#logger = logging.getLogger("pose_match")

# Init the returned tuple
MatchResult = collections.namedtuple("MatchResult", ["match_bool", "error_score", "input_transformation"])

class MatchCombo(object):
    def __init__(self, error_score, input_id, model_id, model_features, input_features, input_transformation):
        self.error_score = error_score
        self.input_id = input_id
        self.model_id = model_id
        self.model_features = model_features # niet noodzaakelijk voor logica, wordt gebruikt voor plotjes
        self.input_features = input_features # same
        self.input_transformation = input_transformation


def single_person(model_features, input_features, normalise=True):

    
    (input_features_copy, model_features_copy) = prepocessing.handle_undetected_points(input_features, model_features)

    if (normalise):
        model_features_copy = normalising.feature_scaling(model_features_copy)
        input_features_copy = normalising.feature_scaling(input_features_copy)

    
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_features_copy)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_features_copy)

    
    
    (input_transformed_face, transformation_matrix_face) = affine_transformation.find_transformation(model_face, input_face)
    (input_transformed_torso, transformation_matrix_torso) = affine_transformation.find_transformation(model_torso, input_torso)
    (input_transformed_legs, transformation_matrix_legs) = affine_transformation.find_transformation(model_legs, input_legs)

    
    input_transformation = prepocessing.unsplit(input_transformed_face, input_transformed_torso, input_transformed_legs)

    
    if(not normalise):
        result = MatchResult(None,
                             error_score=0,
                             input_transformation=input_transformation)
        return result

    max_euclidean_error_face = pose_comparison.max_euclidean_distance(model_face, input_transformed_face)
    max_euclidean_error_torso = pose_comparison.max_euclidean_distance(model_torso, input_transformed_torso)
    max_euclidean_error_legs = pose_comparison.max_euclidean_distance(model_legs, input_transformed_legs)

    max_euclidean_error_shoulders = pose_comparison.max_euclidean_distance_shoulders(model_torso, input_transformed_torso)


    ######### THE THRESHOLDS #######
    eucl_dis_tresh_torso = 0.11 #0.065  of 0.11 ??
    rotation_tresh_torso = 40
    eucl_dis_tresh_legs = 0.055
    rotation_tresh_legs = 40

    eucld_dis_shoulders_tresh = 0.063
    ################################

    result_torso, torso_value = pose_comparison.decide_torso_shoulders_incl(max_euclidean_error_torso, transformation_matrix_torso,
                                                eucl_dis_tresh_torso, rotation_tresh_torso,
                                                max_euclidean_error_shoulders, eucld_dis_shoulders_tresh)

    result_legs = pose_comparison.decide_legs(max_euclidean_error_legs, transformation_matrix_legs,
                                              eucl_dis_tresh_legs, rotation_tresh_legs)

    #TODO: construct a solid score algorithm
    error_score = (max_euclidean_error_torso + max_euclidean_error_legs)/2.0

    result = MatchResult((result_torso and result_legs),
                         error_score=error_score,
                         input_transformation=input_transformation)
    return result, torso_value



def plot_single_person(model_features, input_features, model_image_name, input_image_name, torso_value, input_title = "input",  model_title="model",
                       transformation_title="transformed input -incl. split()"):

    # Filter the undetected features and mirror them in the other pose
    (input_features_copy, model_features_copy) = prepocessing.handle_undetected_points(input_features, model_features)
    
    # plot vars
    markersize = 3

    #Load images
    model_image_name = cv2.resize(model_image_name, (432,368))
    input_image_name = cv2.resize(input_image_name, (432,368))

    # Split features in three parts
    (model_face, model_torso, model_legs) = prepocessing.split_in_face_legs_torso(model_features_copy)
    (input_face, input_torso, input_legs) = prepocessing.split_in_face_legs_torso(input_features_copy)

    # Zoek transformatie om input af te beelden op model
    # Returnt transformatie matrix + afbeelding/image van input op model
    (input_transformed_face, transformation_matrix_face) = affine_transformation.find_transformation(model_face,
                                                                                                     input_face)
    (input_transformed_torso, transformation_matrix_torso) = affine_transformation.find_transformation(model_torso,
                                                                                                       input_torso)
    (input_transformed_legs, transformation_matrix_legs) = affine_transformation.find_transformation(model_legs,
                                                                                                     input_legs)


    whole_input_transform = prepocessing.unsplit(input_transformed_face, input_transformed_torso,
                                                 input_transformed_legs)

    #model_image = plt.imread(model_image_name) #png.read_png_int(model_image_name) #plt.imread(model_image_name)
    #input_image = plt.imread(input_image_name) #png.read_png_int(input_image_name) #plt.imread(input_image_name)

    model_image = draw_humans.draw_humans(model_image_name, model_features, True)  # plt.imread(model_image_name)
    input_image = draw_humans.draw_humans(input_image_name, input_features, True)  # plt.imread(input_image_name)


    input_trans_image = draw_humans.draw_square(model_image_name, model_features)
    input_trans_image = draw_humans.draw_humans(input_trans_image, whole_input_transform,
                                                True)  # plt.imread(input_image_name) png.read_png_int(model_image_name)


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
    implot = ax1.imshow(model_image)
    plt.axis('off')
    #ax1.set_title(model_image_name + ' (model)')
    ax1.set_title(model_title)
    ax1.axis('off')
    # ax1.plot(*zip(*model_features_copy), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    # red_patch = mpatches.Patch(color='magenta', label='model')
    # ax1.legend(handles=[red_patch])

    #ax2.set_title(input_image_name + ' (input)')
    ax2.set_title(input_title)
    ax2.axis('off')
    ax2.imshow(input_image)
    # ax2.plot(*zip(*input_features_copy), marker='o', color='r', ls='', ms=markersize)
    # ax2.legend(handles=[mpatches.Patch(color='red', label='input')])


    ax3.set_title("Score :" + str(torso_value))
    ax3.axis('off')
    ax3.imshow(input_trans_image)
    # ax3.plot(*zip(*model_features_copy), marker='o', color='magenta', ls='', label='model', ms=markersize)  # ms = markersize
    # ax3.plot(*zip(*whole_input_transform), marker='o', color='b', ls='', ms=markersize)
    # ax3.legend(handles=[mpatches.Patch(color='blue', label='transformed input'), mpatches.Patch(color='magenta', label='model')])

    #plot_name = model_image_name.split("/")[-1] + "_" + input_image_name.split("/")[-1]
    plot_name = 'save'
    filename = 'testplots'
    #plt.ion()
    #plt.savefig(filename + '\\' + plot_name + str(random.randint(1,999)) + '.png', bbox_inches='tight')
    #plt.show(block=False)
    return f


