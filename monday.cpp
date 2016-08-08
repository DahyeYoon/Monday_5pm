//
// Created by cvpr on 8/8/16.
//

//
// Created by DahyeYoon on 2/3/16.
//

#include <stdio.h>
#include <stdlib.h>
#include "gr_module_4y.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/cvaux.h"
#include "opencv/cxcore.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <time.h>

#define max_peoples 10
#define max_classss 10
#define frame_ccc 16

#define bDEBUG 1

using namespace cv;
using namespace std;

int sum_factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n + sum_factorial(n - 1);
}

gr_module_4y::gr_module_4y(void) {


}

gr_module_4y::~gr_module_4y(void) {

}


int gr_module_4y::check_roi(int min, int max, int value) {

    int result = value;
    if (value < 0) {
        result = 0;
    }
    if (value > max) {
        result = max;
    }
    return result;

}

bool gr_module_4y::inroi(CvPoint roi_lt, CvPoint roi_rb, float mean_x, float mean_y) {

    if (roi_lt.x <= mean_x && roi_rb.x >= mean_x && roi_lt.y <= mean_y && roi_rb.y >= mean_y) {
        return true;
    }
    return false;

}

CvPoint gr_module_4y::max_roi(CvPoint input[frame_ccc], int human_idx) {
    CvPoint max = {0, 0};
    if (human_turn[human_idx]) {
        for (int i = 0; i < frame_ccc; i++) {
            if (input[i].x >= max.x) {

            }
            if (input[i].y >= max.y) {
                max.y = input[i].y;
            }
        }
    }
    else {
        for (int i = 0; i <= human_frame_h[human_idx]; i++) {
            if (input[i].x >= max.x) {
                max.x = input[i].x;
            }
            if (input[i].y >= max.y) {
                max.y = input[i].y;
            }
        }
    }
    return max;
}

CvPoint gr_module_4y::min_roi(CvPoint input[frame_ccc], int human_idx) {
    CvPoint min = {1000, 1000};
    if (human_turn[human_idx]) {
        for (int i = 0; i < frame_ccc; i++) {
            if (input[i].x <= min.x) {
                min.x = input[i].x;
            }
            if (input[i].y <= min.y) {
                min.y = input[i].y;
            }
        }
    }
    else {
        for (int i = 0; i <= human_frame_h[human_idx]; i++) {
            if (input[i].x <= min.x) {
                min.x = input[i].x;
            }
            if (input[i].y <= min.y) {
                min.y = input[i].y;
            }
        }
    }
    return min;

}
//commit

void gr_module_4y::init(char path[]) {
    //cout<<"intialization~!~!"<<endl;
    max_nr_attr = 64;
    init_counter = 0;
    max_people = 10;
    max_class = 10;
    frame_con = 16;
    show_track = 1;
    test_bool = true;
    roi_hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
    pose_dis = Malloc(double, max_class);
    motion_dis = Malloc(double, max_class);
    global_walk_sum = 0;
    InitTrackInfo(&trackInfo, track_length, init_gap);
    InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);
    InitDescInfo(&pcInfo, 8, false, patch_size, nxy_cell, nt_cell);

    for (int i = 0; i < max_peoples; i++) {
        first_f_bool[i] = true;
        second_f_bool[i] = false;
        circle_hog[i] = 0;
        moving_jug[i] = false;
    }
    hog_weight[0] = 9;
    hog_weight[1] = 9;
    hog_weight[2] = 9;
    hog_weight[3] = 7;
    hog_weight[4] = 5;
    hog_weight[5] = 0;
    hog_weight[6] = 0;
    hog_weight[7] = 8;
    hog_weight[8] = 4;
    hog_weight[9] = 7;

    hog_factorial = sum_factorial(hogInfo.dim * 3);
    hof_factorial = sum_factorial(hofInfo.dim * 3);
    mbh_factorial = sum_factorial(mbhInfo.dim * 3);


    FILE *f_in;
//    sprintf(dat_path, "/home/cvpr/repository/database/what" );
    strcpy(dat_path, path);
    char svmpath[100];

    dim = sum_factorial(hogInfo.dim * 3) * 3 + sum_factorial(hofInfo.dim * 3);

//	dest_all = cvCreateMat(dim,100,CV_32FC1);
    time_t startTime, endTime;
    float processtime;
    startTime =clock();

    char model_file_name[1024];
//    sprintf(model_file_name, "%s/hog_classifier/detcom_model.model", dat_path);
    sprintf(model_file_name, "%s/hog_classifier/detcom_model.model", dat_path);
    //printf("load detcom\n");
    if ((motion_model = load_model(model_file_name)) == 0) {
        printf("can't open detcom model file!\n");
        //exit(1);
    }
    sprintf(model_file_name, "%s/hog_classifier/hri_pose_quar.model", dat_path);
    //printf("load hog\n");
    if ((pose_model = load_model(model_file_name)) == 0) {
        printf("can't open pose model file!\n");
        //exit(1);
    }
    endTime = clock();
    processtime = ((float)(endTime-startTime));
//    cout<<"MODEL LOAD TIME: "<<processtime<<endl;
    printf("Model Load Time: %f\n", processtime);
    motion_x = (struct feature_node *) malloc(602 * sizeof(struct feature_node));
    pose_x = (struct feature_node *) malloc(8822 * sizeof(struct feature_node));
//	cout<<"Initialization Complete!!!"<<endl;


// ========================================DETCOM - PCA Part===========================================
    ifstream DETCOM_EigenV;
    float *detcom_eigen = (float *) malloc(11912400 * sizeof(float));
    DETCOM_EigenV.open("/home/cvpr/repository/database/what/Eigen_vector/DETCOM_train_Eigen.txt");
    for (int i = 0; i < 11912400; i++)
        DETCOM_EigenV >> detcom_eigen[i];

    if (!DETCOM_EigenV.is_open())
        return;
    DETCOM_Eigen_mat = cvCreateMat(600, 19854, CV_32FC1);
    cvSetData(DETCOM_Eigen_mat, detcom_eigen, DETCOM_Eigen_mat->step);
    //cout<<"==============EIGEN_VECTOR:================="<<cvGet2D(DETCOM_Eigen_mat,0,0).val[0]<<endl;
    //cout<<"==============EIGEN_VECTOR2:================="<<cvGet2D(DETCOM_Eigen_mat,0,1).val[0]<<endl;
    //cout<<"==============EIGEN_VECTOR2:================="<<cvGet2D(DETCOM_Eigen_mat,1,0).val[0]<<endl;
//    CvMat *DETCOM_Eig_Trans = cvCreateMat(19854, 600, CV_32FC1);
//    cvTranspose(DETCOM_Eigen_mat, DETCOM_Eig_Trans);
//    cvReleaseMat(&DETCOM_Eigen_mat);

// ====================================================================================================


}

void gr_module_4y::DETCOM_feature(Mat input_mat, CvMat *output_mat, Mat &all_mat, int start_point) {
    //cout<<"DETCOM_feature IN: "<<frame_num<<endl;
////////// covariance part /////////////
    double cov_temp = 0.0;
    CvMat *dt_mat = cvCreateMat(input_mat.size().height, input_mat.size().width, CV_32FC1);
    CvMat *dt_mat_t = cvCreateMat(input_mat.size().width, input_mat.size().height, CV_32FC1);
    CvMat *num_mat = cvCreateMat(input_mat.size().width, input_mat.size().width, CV_32FC1);
    CvMat *cov_result = cvCreateMat(input_mat.size().width, input_mat.size().width, CV_32FC1);
    for (unsigned j = 0; (j < input_mat.cols); j++) // mean minus
    {
        double mean_value = mean(input_mat.col(j))[0];
        for (int i = 0; i < input_mat.rows; i++) {
            cov_temp = input_mat.at<float>(i, j) - mean_value;
            cvmSet(dt_mat, i, j, cov_temp);
        }
    }
    for (unsigned j = 0; (j < input_mat.size().width); j++) // diag(eigen_val) // diagonal matrix
    {
        for (int i = 0; i < input_mat.size().width; i++) {
            cvmSet(num_mat, i, j, input_mat.size().height);
        }
    }
    cvTranspose(dt_mat, dt_mat_t);
    cvMatMul(dt_mat_t, dt_mat, cov_result);
    cvDiv(cov_result, num_mat, cov_result); // (x-m)(y-m)/n

    cvReleaseMat(&dt_mat);
    cvReleaseMat(&dt_mat_t);
    cvReleaseMat(&num_mat);

    ///////// eigen decomposition ///////////
    Mat cov_result_mat = cov_result;
    Mat eigen_vec, eigen_val;
    eigen(cov_result_mat, true, eigen_val, eigen_vec); /// sigma = V*D* V';
    Mat eigen_val_log = eigen_val.clone();
    log(eigen_val_log, eigen_val_log);  /// D^ = log(D)

    CvMat *eigen_val_c = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *eigen_vec_c = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *eigen_vec_ct = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    cvSetZero(eigen_vec_c);
    cvSetZero(eigen_vec_ct);
    cvSetZero(eigen_val_c);
    for (int i = 0; i < eigen_val_log.size().height; i++) // diag(D') // diagonal matirx
    {
        //	cout<<eigen_val_log.at<float>(i,0)<<endl;
        cvmSet(eigen_val_c, i, i, eigen_val_log.at<float>(i, 0));
    }
    for (int i = 0; i < eigen_vec.size().height; i++) // mat to cvmat
    {
        for (int j = 0; j < eigen_vec.size().width; j++) {
            cvmSet(eigen_vec_c, i, j, eigen_vec.at<float>(i, j));
        }
    }
    cvTranspose(eigen_vec_c, eigen_vec_ct); // V'

    CvMat *result_temp2 = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *result = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    cvMatMul(eigen_vec_ct, eigen_val_c, result_temp2);
    cvMatMul(result_temp2, eigen_vec_c, result);        // sigma^ = V*D^*V'

    int temp_n = 0, temp_n2 = start_point;

    for (int i = 0; i < eigen_vec.size().height; i++) {
        for (int j = 0; j < i + 1; j++) {
            cvmSet(output_mat, temp_n++, 0, cvmGet(result, i, j));
            //cvmSet(all_mat,temp_n2++,0,cvmGet(result,i,j));
            all_mat.at<float>(temp_n2++, 0) = cvmGet(result, i, j);
            //fprintf(f_f,"%.7f\n",cvmGet(result,i,j));
            //fprintf(f_all,"%.7f\n",cvmGet(result,i,j));
            //			cvmSet(output_mat,temp_n++,0,cvmGet(result,j,j+i));
            //			all_mat.at<float>(temp_n2++,0) = cvmGet(result,j,j+i);
        }
    }
    //cvCopy(descriptor,output_mat);
/*	et=clock();
	gt=(float)(et-st)/(CLOCKS_PER_SEC);
	printf("vector : %f\n",gt);*/
    eigen_val_log.release();
    eigen_val.release();
    eigen_vec.release();

    cvReleaseMat(&result);
    cvReleaseMat(&result_temp2);

    cvReleaseMat(&eigen_val_c);
    cvReleaseMat(&eigen_vec_ct);
    cvReleaseMat(&eigen_vec_c);
    //cout<<"DETCOM_feature OUT: "<<frame_num<<endl;
}

void gr_module_4y::DETCOM_feature(Mat input_mat, CvMat *output_mat, Mat &all_mat, int start_point, int colums) {
    //cout<<"DETCOM_feature2 IN: "<<frame_num<<endl;
    ////////// covariance part /////////////
    double cov_temp = 0.0;
    CvMat *dt_mat = cvCreateMat(input_mat.size().height, input_mat.size().width, CV_32FC1);
    CvMat *dt_mat_t = cvCreateMat(input_mat.size().width, input_mat.size().height, CV_32FC1);
    CvMat *num_mat = cvCreateMat(input_mat.size().width, input_mat.size().width, CV_32FC1);
    CvMat *cov_result = cvCreateMat(input_mat.size().width, input_mat.size().width, CV_32FC1);
    for (unsigned j = 0; (j < input_mat.cols); j++) // mean minus
    {
        double mean_value = mean(input_mat.col(j))[0];
        for (int i = 0; i < input_mat.rows; i++) {
            cov_temp = input_mat.at<float>(i, j) - mean_value;
            cvmSet(dt_mat, i, j, cov_temp);
        }
    }
    for (unsigned j = 0; (j < input_mat.size().width); j++) // diag(eigen_val) // diagonal matrix
    {
        for (int i = 0; i < input_mat.size().width; i++) {
            cvmSet(num_mat, i, j, input_mat.size().height);
        }
    }
    cvTranspose(dt_mat, dt_mat_t);
    cvMatMul(dt_mat_t, dt_mat, cov_result);
    cvDiv(cov_result, num_mat, cov_result); // (x-m)(y-m)/n

    cvReleaseMat(&dt_mat);
    cvReleaseMat(&dt_mat_t);
    cvReleaseMat(&num_mat);

    ///////// eigen decomposition ///////////
    Mat cov_result_mat = cov_result;
    Mat eigen_vec, eigen_val;
    eigen(cov_result_mat, true, eigen_val, eigen_vec); /// sigma = V*D* V';
    Mat eigen_val_log = eigen_val.clone();
    log(eigen_val_log, eigen_val_log);  /// D^ = log(D)

    CvMat *eigen_val_c = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *eigen_vec_c = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *eigen_vec_ct = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    cvSetZero(eigen_vec_c);
    cvSetZero(eigen_vec_ct);
    cvSetZero(eigen_val_c);
    for (int i = 0; i < eigen_val_log.size().height; i++) // diag(D') // diagonal matirx
    {
        //	cout<<eigen_val_log.at<float>(i,0)<<endl;
        cvmSet(eigen_val_c, i, i, eigen_val_log.at<float>(i, 0));
    }
    for (int i = 0; i < eigen_vec.size().height; i++) // mat to cvmat
    {
        for (int j = 0; j < eigen_vec.size().width; j++) {
            cvmSet(eigen_vec_c, i, j, eigen_vec.at<float>(i, j));
        }
    }
    cvTranspose(eigen_vec_c, eigen_vec_ct); // V'

    CvMat *result_temp2 = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *result = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    cvMatMul(eigen_vec_ct, eigen_val_c, result_temp2);
    cvMatMul(result_temp2, eigen_vec_c, result);        // sigma^ = V*D^*V'

    int temp_n = 0, temp_n2 = start_point;

    for (int i = 0; i < eigen_vec.size().height; i++) {
        for (int j = 0; j < i + 1; j++) {
            cvmSet(output_mat, temp_n++, 0, cvmGet(result, i, j));
            //cvmSet(all_mat,temp_n2++,colums,cvmGet(result,i,j));
            all_mat.at<float>(temp_n2++, colums) = cvmGet(result, i, j);
            //fprintf(f_f,"%.7f\n",cvmGet(result,i,j));
            //fprintf(f_all,"%.7f\n",cvmGet(result,i,j));
            //			cvmSet(output_mat,temp_n++,0,cvmGet(result,j,j+i));
            //			all_mat.at<float>(temp_n2++,0) = cvmGet(result,j,j+i);
        }
    }
    //cvCopy(descriptor,output_mat);
/*	et=clock();
	gt=(float)(et-st)/(CLOCKS_PER_SEC);
	printf("vector : %f\n",gt);*/
    eigen_val_log.release();
    eigen_val.release();
    eigen_vec.release();

    cvReleaseMat(&result);
    cvReleaseMat(&result_temp2);

    cvReleaseMat(&eigen_val_c);
    cvReleaseMat(&eigen_vec_ct);
    cvReleaseMat(&eigen_vec_c);
    //cout<<"DETCOM_feature2 OUT: "<<frame_num<<endl;
}

void gr_module_4y::DETCOM_feature_w(Mat input_mat, CvMat *output_mat, Mat &all_mat, int start_point, FILE *f_all) {
    //cout<<"DETCOM_feature_w IN: "<<frame_num<<endl;
    ////////// covariance part /////////////
    double cov_temp = 0.0;
    CvMat *dt_mat = cvCreateMat(input_mat.size().height, input_mat.size().width, CV_32FC1);
    CvMat *dt_mat_t = cvCreateMat(input_mat.size().width, input_mat.size().height, CV_32FC1);
    CvMat *num_mat = cvCreateMat(input_mat.size().width, input_mat.size().width, CV_32FC1);
    CvMat *cov_result = cvCreateMat(input_mat.size().width, input_mat.size().width, CV_32FC1);
    for (unsigned j = 0; (j < input_mat.cols); j++) // mean minus
    {
        double mean_value = mean(input_mat.col(j))[0];
        for (int i = 0; i < input_mat.rows; i++) {
            cov_temp = input_mat.at<float>(i, j) - mean_value;
            cvmSet(dt_mat, i, j, cov_temp);
        }
    }
    for (unsigned j = 0; (j < input_mat.size().width); j++) // diag(eigen_val) // diagonal matrix
    {
        for (int i = 0; i < input_mat.size().width; i++) {
            cvmSet(num_mat, i, j, input_mat.size().height);
        }
    }
    cvTranspose(dt_mat, dt_mat_t);
    cvMatMul(dt_mat_t, dt_mat, cov_result);
    cvDiv(cov_result, num_mat, cov_result); // (x-m)(y-m)/n

    cvReleaseMat(&dt_mat);
    cvReleaseMat(&dt_mat_t);
    cvReleaseMat(&num_mat);

    ///////// eigen decomposition ///////////
    Mat cov_result_mat = cov_result;
    Mat eigen_vec, eigen_val;
    eigen(cov_result_mat, true, eigen_val, eigen_vec); /// sigma = V*D* V';
    Mat eigen_val_log = eigen_val.clone();
    log(eigen_val_log, eigen_val_log);  /// D^ = log(D)

    CvMat *eigen_val_c = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *eigen_vec_c = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *eigen_vec_ct = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    cvSetZero(eigen_vec_c);
    cvSetZero(eigen_vec_ct);
    cvSetZero(eigen_val_c);
    for (int i = 0; i < eigen_val_log.size().height; i++) // diag(D') // diagonal matirx
    {
        //	cout<<eigen_val_log.at<float>(i,0)<<endl;
        cvmSet(eigen_val_c, i, i, eigen_val_log.at<float>(i, 0));
    }
    for (int i = 0; i < eigen_vec.size().height; i++) // mat to cvmat
    {
        for (int j = 0; j < eigen_vec.size().width; j++) {
            cvmSet(eigen_vec_c, i, j, eigen_vec.at<float>(i, j));
        }
    }
    cvTranspose(eigen_vec_c, eigen_vec_ct); // V'

    CvMat *result_temp2 = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    CvMat *result = cvCreateMat(eigen_val.size().height, eigen_val.size().height, CV_32FC1);
    cvMatMul(eigen_vec_ct, eigen_val_c, result_temp2);
    cvMatMul(result_temp2, eigen_vec_c, result);        // sigma^ = V*D^*V'

    int temp_n = 0, temp_n2 = start_point;

    for (int i = 0; i < eigen_vec.size().height; i++) {
        for (int j = 0; j < i + 1; j++) {
            //fprintf(f_des,"%.7f\n",cvmGet(result,i,j));
            cvmSet(output_mat, temp_n++, 0, cvmGet(result, i, j));
            all_mat.at<float>(temp_n2++, 0) = cvmGet(result, i, j);
            fprintf(f_all, "%.7f\n", cvmGet(result, i, j));
            //			cvmSet(output_mat,temp_n++,0,cvmGet(result,j,j+i));
            //			all_mat.at<float>(temp_n2++,0) = cvmGet(result,j,j+i);
        }
    }

/*	for(int i = 0 ; i < eigen_vec.size().height; i++)
	{
		for(int j = 0 ; j < eigen_vec.size().height-i; j++)
		{
			//cvmSet(output_mat,output_idx,temp_n++,cvmGet(result,j,j+i));
//			cout<<j+i<<endl;
			cvmSet(output_mat,temp_n++,0,cvmGet(result,i,j));
			all_mat.at<float>(temp_n2++,0) = cvmGet(result,i,j);
			fprintf(f_all,"%.7f\n",cvmGet(result,i,j));
		}
	}*/
    //cvCopy(descriptor,output_mat);
/*	et=clock();
	gt=(float)(et-st)/(CLOCKS_PER_SEC);
	printf("vector : %f\n",gt);*/
    eigen_val_log.release();
    eigen_val.release();
    eigen_vec.release();

    cvReleaseMat(&result);
    cvReleaseMat(&result_temp2);

    cvReleaseMat(&eigen_val_c);
    cvReleaseMat(&eigen_vec_ct);
    cvReleaseMat(&eigen_vec_c);
    //cout<<"DETCOM_feature_w OUT: "<<frame_num<<endl;
}

int gr_module_4y::do_hri_motion_predict(Mat &all_mat) {
    clock_t begin, end;
    begin = clock();

//    cout<<"do hri_motion_predict method IN: "<<frame_num<<endl;
    int predict_label;
    int nr_class = get_nr_class(motion_model);

    double *prob_estimates = NULL;
    int j, n;
    int nr_feature = get_nr_feature(motion_model);
    //cout<<"~~~~~~~~~~~1 :"<<nr_feature<<endl;
    if (motion_model->bias >= 0) {
        cout << "We use bias" << endl;
        n = nr_feature + 1;
    }
    else
        n = nr_feature;

    int feature_dim[] = {sum_factorial(96) * 3 + sum_factorial(108), sum_factorial(96), sum_factorial(108),
                         sum_factorial(96), sum_factorial(96)};
    int feature_index = 0;
    int nImgNum = 1;
    float buf;
    int k = 1;

    int i;
    //while (k * n >= max_nr_attr - 2)    // need one more for index = -1
    //{
    // max_nr_attr *= 2;
//        motion_x = (struct feature_node *) realloc(motion_x, n * sizeof(struct feature_node));
//        if (motion_x == NULL) {
//            printf("Motion x reallocation failure!\n");
//            exit(0);
//        }
    //}
    for (i = 0; i < nr_feature; i++) {
        motion_x[i].index = i + 1;
        motion_x[i].value = all_mat.at<float>(0, i);
        //cout<<"============= motion_x[] ===============:"<<motion_x[i].value<<endl;

    }
    if (motion_model->bias >= 0) {
        motion_x[i].index = n;
        motion_x[i].value = motion_model->bias;
        i++;
    }
    motion_x[i].index = -1;

    if (flag_predict_probability) {
        int j;
        predict_label = predict_probability(motion_model, motion_x, prob_estimates);
    }
    else {
        predict_label = predict(motion_model, motion_x);
    }
    return predict_label;

    //cout<<"do hri_motion_predict method OUT: "<<frame_num<<endl;
    end = clock();

    printf("!!!!!!!!! Time!!!!!!!!!: %f\n", ((double)(end-begin))/CLOCKS_PER_SEC);


}

int gr_module_4y::do_hri_motion_predict(Mat &all_mat, double *dis) {
//    cout<<"do hri_motion_predict method IN+dis: "<<frame_num<<endl;

    clock_t start, end;
    float processT;
    start = clock();




    int predict_label;
    int nr_class = get_nr_class(motion_model);

    double *prob_estimates = NULL;
    int j, n;
    int nr_feature = get_nr_feature(motion_model);
    //cout<<"~~~~~~~~~~~2:"<<nr_feature<<endl;

    if (motion_model->bias >= 0) {
        //cout << "We use bias" << endl;
        n = nr_feature + 1;
    }
    else
        n = nr_feature;

    //int feature_dim[] = {sum_factorial(96) * 3 + sum_factorial(108), sum_factorial(96), sum_factorial(108),
    //sum_factorial(96), sum_factorial(96)};
    int feature_index = 0;
    int nImgNum = 1;
    float buf;
    int k = 1;

    int i;
//    while (k * n >= max_nr_attr - 2)    // need one more for index = -1
//    {
//        max_nr_attr *= 2;

//    cout << "allocate memory directly instead of using max_nr_attr: " << max_nr_attr << endl;

    //  motion_x = (struct feature_node *) realloc(motion_x, (n+1) * sizeof(struct feature_node));

//        if (motion_x == NULL) {
//            printf("Motion x reallocation failure!\n");
//            exit(0);
//        } else{
//            cout << "size of motion_x: " << n+1 << endl;
//        }
    //}

    for (i = 0; i < nr_feature; i++) {
        motion_x[i].index = i + 1;
        motion_x[i].value = all_mat.at<float>(0, i);
    }

    //cout << "assign bias term" << endl;

    if (motion_model->bias >= 0) {
        motion_x[i].index = n;
        motion_x[i].value = motion_model->bias;
        i++;
    }

//    cout << "the element before the last of motion_x: " << motion_x[i-2].index << ", " << motion_x[i-2].value << endl;
//    cout << "the last element of motion_x: " << motion_x[i-1].index << ", " << motion_x[i-1].value << endl;
//    cout << "reset the last element motion_x[" << i << "].index: "<< motion_x[i].index << endl << endl;
    motion_x[i].index = -1;

    if (flag_predict_probability) {
        int j;
        predict_label = predict_probability(motion_model, motion_x, prob_estimates);
    }
    else {
//		cout<<"in function predict!!"<<endl;
        predict_label = predict_with_dis(motion_model, motion_x, dis);
        //cout<<"!!!!!!!!!!!!!!!!!!!!!dis : "<<dis[0]<<endl;
    }

    end = clock();
    processT=((double(end-start)));

    cout<<"TIME : "<<processT<<endl;

    return predict_label;
    //cout<<"do hri_motion_predict method OUT+dis: "<<frame_num<<endl;




}


int gr_module_4y::do_hri_pose_predict(Mat &all_mat) {

    //cout<<"do hri_motion_pose method IN: "<<frame_num<<endl;
    int predict_label;
    int nr_class = get_nr_class(pose_model);
    double *prob_estimates = NULL;
    int j, n;
    int nr_feature = get_nr_feature(pose_model);
    if (pose_model->bias >= 0)
        n = nr_feature + 1;
    else
        n = nr_feature;

    int feature_dim[] = {sum_factorial(96) * 3 + sum_factorial(108), sum_factorial(96), sum_factorial(108),
                         sum_factorial(96), sum_factorial(96)};
    int feature_index = 0;
    int nImgNum = 1;
    float buf;
    int k = 1;

    int i;
//    while (k * n >= max_nr_attr - 2)    // need one more for index = -1
//    {
//        max_nr_attr *= 2;
//        pose_x = (struct feature_node *) realloc(pose_x, max_nr_attr * sizeof(struct feature_node));
//        if (pose_x == NULL) {
//            printf("Pose x reallocation failure!\n");
//            exit(0);
//        }
//    }
    for (i = 0; i < nr_feature; i++) {
        pose_x[i].index = i + 1;
        pose_x[i].value = all_mat.at<float>(0, i);
    }
    if (pose_model->bias >= 0) {
        pose_x[i].index = n;
        pose_x[i].value = pose_model->bias;
        i++;
    }
    pose_x[i].index = -1;

    if (flag_predict_probability) {
        int j;
        predict_label = predict_probability(pose_model, pose_x, prob_estimates);
    }
    else {
        predict_label = predict(pose_model, pose_x);
    }
    return predict_label;
    //cout<<"do hri_motion_pose method out: "<<frame_num<<endl;

}

int gr_module_4y::do_hri_pose_predict(Mat &all_mat, double *dis) {

    //cout<<"do hri_motion_pose method IN+dis: "<<frame_num<<endl;
    int predict_label;
    int nr_class = get_nr_class(pose_model);
    double *prob_estimates = NULL;
    int j, n;
    int nr_feature = get_nr_feature(pose_model);
    if (pose_model->bias >= 0)
        n = nr_feature + 1;
    else
        n = nr_feature;

    int feature_dim[] = {sum_factorial(96) * 3 + sum_factorial(108), sum_factorial(96), sum_factorial(108),
                         sum_factorial(96), sum_factorial(96)};
    int feature_index = 0;
    int nImgNum = 1;
    float buf;
    int k = 1;

    int i;
//    while (k * n >= max_nr_attr - 2)    // need one more for index = -1
//    {
//        max_nr_attr *= 2;
//        cout << endl << "max_nr_attr value: " << max_nr_attr << endl;
//        pose_x = (struct feature_node *) realloc(pose_x, max_nr_attr * sizeof(struct feature_node));
//        if (pose_x == NULL) {
//            printf("Pose x reallocation failure!\n");
//            exit(0);
//        }
//    }
    for (i = 0; i < nr_feature; i++) {
        pose_x[i].index = i + 1;
        pose_x[i].value = all_mat.at<float>(0, i);
    }
    if (pose_model->bias >= 0) {
        pose_x[i].index = n;
        pose_x[i].value = pose_model->bias;
        i++;
    }
    pose_x[i].index = -1;

    if (flag_predict_probability) {
        int j;
        predict_label = predict_probability(pose_model, pose_x, prob_estimates);
    }
    else {
        predict_label = predict_with_dis(pose_model, pose_x, dis);
    }
    return predict_label;
    //cout<<"do hri_motion_pose method OUT+dis: "<<frame_num<<endl;
}

int gr_module_4y::determin_label(float hog_label[frame_ccc], float hog_dis[max_classss][frame_ccc], float s_label,
                                 float s_dis[max_classss],
                                 const int hog_weight[max_classss]) { // determining predict number of class
    //cout<<"determin_label IN: "<<frame_num<<endl;
    float last_label[max_classss * 2], last_dis[max_classss * 2];
    float detmer_dis[max_classss];
    float max_label = -50, max_dist = -50, hog_max_dist = -50, stip_max_dist = -50;
    for (int _j = 0; _j < max_classss * 2; _j++) {
        if (_j < max_classss) {
            float temp_l = 0, temp_d = 0;
            for (int _j2 = 0; _j2 < frame_ccc; _j2++) {
                //			cout<<"hog_label["<<_j<<"]["<<_j2<<"] : "<<hog_label[_j][_j2]<<endl;
                //	temp_l += hog_label[_j2];
                temp_d += hog_dis[_j][_j2];
            }
            //		cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
            //	last_label[_j] = temp_l/frame_con;
            last_dis[_j] = temp_d / frame_ccc;
        }
        else {
            //	last_label[_j] = s_label[_j-max_class];
            last_dis[_j] = s_dis[_j - max_classss];
        }
    }

    for (int _j = 0; _j < max_classss; _j++) {
        double h_wei = ((double) hog_weight[_j] / (double) 10);
        //		cout<<hog_weight<<","<<h_wei<<","<<abs(1-h_wei)<<endl;
        double h_val = (double) (h_wei * (max_classss + last_dis[_j]));
        /*		if(h_val>max_class+1){h_val = h_val + 1;}*/
        double b_val = (double) (abs(1 - h_wei) * (max_classss + last_dis[_j + max_classss]));
        /*		if(b_val>max_class+1){b_val = b_val + 1;}*/

        detmer_dis[_j] = h_val + b_val; //0.1~0.9 weight
        /*	if(last_label[_j]>=0.5)
		{
			detmer_dis[_j] += last_label[_j];
		}*/
        if (max_dist < detmer_dis[_j]) {
            max_dist = detmer_dis[_j];
            max_label = _j + 1;
        }
    }
    return max_label;
    //cout<<"determin_label OUT: "<<frame_num<<endl;
}

int gr_module_4y::determin_label(float hog_label[frame_ccc], float hog_dis[max_classss][frame_ccc], float s_label,
                                 float s_dis[max_classss], const int hog_weight[max_classss],
                                 bool moving) { // determining predict number of class
    //cout<<"determin_label2 IN: "<<frame_num<<endl;
    float last_label[max_classss * 2], last_dis[max_classss * 2];
    float detmer_dis[max_classss];
    float max_label = -50, max_dist = -50, hog_max_dist = -50, stip_max_dist = -50;
    for (int _j = 0; _j < max_classss * 2; _j++) {
        if (_j < max_classss) {
            float temp_l = 0, temp_d = 0;
            for (int _j2 = 0; _j2 < frame_ccc; _j2++) {
                //			cout<<"hog_label["<<_j<<"]["<<_j2<<"] : "<<hog_label[_j][_j2]<<endl;
                //	temp_l += hog_label[_j2];
                temp_d += hog_dis[_j][_j2];
                //			cout<<_j<<" : "<<_j2<<", hog_dis[_j][_j2] : "<<hog_dis[_j][_j2]<<endl;
            }
            //		cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
            //	last_label[_j] = temp_l/frame_con;
            last_dis[_j] = temp_d / frame_ccc;
            //		cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
        }
        else {
            //	last_label[_j] = s_label[_j-max_classss];
            last_dis[_j] = s_dis[_j - max_classss];
        }
    }


    if (!moving) {
        last_dis[5] = -50;
        last_dis[6] = -50;
    }
    else {
        last_dis[5] = -10;
        last_dis[6] = -10;
    }

    for (int _j = 0; _j < max_classss; _j++) {

        double h_wei = ((double) hog_weight[_j] / (double) 10);
        //	double h_val = (double)(h_wei*(max_classss+last_dis[_j]));
        double b_val = (double) (abs(1 - h_wei) * (max_classss + last_dis[_j + max_classss]));
        double h_val = 0;
        if (_j == 0 || _j == 1) {
            h_val = (double) ((h_wei + 0.05) * (max_classss + last_dis[_j]));
        }
        else {
            h_val = (double) (h_wei * (max_classss + last_dis[_j]));
        }
        detmer_dis[_j] = h_val + b_val; //0.1~0.9 weight
        /*	if(last_label[_j]>=0.5)
		{
			detmer_dis[_j] += last_label[_j];
		}*/
//		cout<<"determin_pair_label_i-"<<_j<<", h_val : "<<h_val<<", b_val : "<<b_val<<" , detmer_dis[_j] : "<<detmer_dis[_j]<<endl;
        if (max_dist < detmer_dis[_j]) {
            max_dist = detmer_dis[_j];
            max_label = _j + 1;
        }
    }
    return max_label;
    //cout<<"determin_label2 OUT: "<<frame_num<<endl;
}

int gr_module_4y::determin_label(float hog_label[frame_ccc], float hog_dis[max_classss][frame_ccc], float s_label,
                                 float s_dis[max_classss], const int hog_weight[max_classss], bool moving,
                                 double sum_distance, double sum_width,
                                 double sum_height) { // determining predict number of class
    //cout<<"determin_label3 IN: "<<frame_num<<endl;
    //cout << "output" << endl;
    float last_label[max_class * 2], last_dis[max_class * 2];
    float detmer_dis[max_class];
    float max_label = -50, max_dist = -50, hog_max_dist = -50, stip_max_dist = -50;
    for (int _j = 0; _j < max_class * 2; _j++) {
        if (_j < max_class) {
            float temp_l = 0, temp_d = 0;
            for (int _j2 = 0; _j2 < frame_con; _j2++) {
                //			cout<<"hog_label["<<_j<<"]["<<_j2<<"] : "<<hog_label[_j][_j2]<<endl;
                //	temp_l += hog_label[_j2];
                temp_d += hog_dis[_j][_j2];
                //			cout<<_j<<" : "<<_j2<<", hog_dis[_j][_j2] : "<<hog_dis[_j][_j2]<<endl;
            }
            //		cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
            //	last_label[_j] = temp_l/frame_con;
            last_dis[_j] = temp_d / frame_con;
            //		cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
        }
        else {
            //	last_label[_j] = s_label[_j-max_class];
            last_dis[_j] = s_dis[_j - max_class];
        }
    }


    if (!moving) {
        //cout << "is no moving??" << endl;
        last_dis[5 + max_class] = (last_dis[5 + max_class] > 0 ? 1 : -1) * (abs(last_dis[5 + max_class]) + 1);
        last_dis[6 + max_class] = (last_dis[6 + max_class] > 0 ? 1 : -1) * (abs(last_dis[6 + max_class]) + 1);
//		last_dis[8+max_class] -= 0.8;
//		last_dis[3] -= 1;
//		last_dis[7] -= 0.8;
    }
    else {
//		last_dis[7+max_class] -= 0.1;
//		last_dis[3] -= 1;
        //	last_dis[2] -= 2;
        if ((sum_distance >= 150 && sum_distance < 330) &&
            ((sum_width >= 180 && sum_width <= 400) && (sum_height >= 200 && sum_height <= 400))) {
            //cout << "is falling?" << endl;
            last_dis[5 + max_class] = (last_dis[5 + max_class] > 0 ? 1 : -1) * (abs(last_dis[5 + max_class]) + 1);
            last_dis[6 + max_class] = (last_dis[6 + max_class] > 0 ? 1 : -1) * (abs(last_dis[6 + max_class]) + 1);
            last_dis[8 + max_class] += 0.8;
        }
        else {
            last_dis[6 + max_class] -= 0.8;
            last_dis[8 + max_class] -= 0.8;
        }
        /*	last_dis[3] -= 1;
		if( sum_distance < 200)
		{
			last_dis[6+max_class] -= 1;
		}

		else if(sum_distance >= 100 && sum_width >= 150 && sum_width <= 400 && sum_height >= 150)
		{
			last_dis[8+max_class] += 2;
		}
		else
		{
			last_dis[6+max_class] -= 1;
		}	*/

    }

    for (int _j = 0; _j < max_class; _j++) {
//		double b_val = 0;
        double h_wei = ((double) hog_weight[_j] / (double) 10);
        //		cout<<hog_weight<<","<<h_wei<<","<<abs(1-h_wei)<<endl;
        //double h_val = (double)(h_wei*(max_class+last_dis[_j]));
        double h_val = 0;
//		if(_j == 8 ||_j == 7)
        {
//			h_val = (double)((h_wei+0.03)*(max_class+last_dis[_j]));
        }
//		else
        {
            h_val = (double) ((h_wei) * (max_class + last_dis[_j]));
        }

        /*		if(h_val>max_class+1){h_val = h_val + 1;}*/
        double b_val = (double) (abs(1 - h_wei) * (max_class + last_dis[_j + max_class]));
        /*if(_j == 3 ||_j == 4)
		{
			b_val = (double)(abs(1-h_wei-0.1)*(max_class+last_dis[_j+max_class]));
		}
		else
		{
			b_val = (double)(abs(1-h_wei)*(max_class+last_dis[_j+max_class]));
		}*/

        /*		if(b_val>max_class+1){b_val = b_val + 1;}*/

        detmer_dis[_j] = h_val + b_val; //0.1~0.9 weight
        /*	if(last_label[_j]>=0.5)
		{
			detmer_dis[_j] += last_label[_j];
		}*/
        //cout << "determin_pair_label_i-" << _j << ", h_val : " << h_val << ", b_val : " << b_val
        //<< " , detmer_dis[_j] : " << detmer_dis[_j] << endl;
        if (max_dist < detmer_dis[_j]) {
            max_dist = detmer_dis[_j];
            max_label = _j + 1;
        }

    }
    //cout << "max_label : " << max_label << " , max_dist : " << max_dist << endl;
    return max_label;
    //cout<<"determin_label3 OUT: "<<frame_num<<endl;
}

int gr_module_4y::determin_label2(float hog_label[frame_ccc], float hog_dis[max_classss][frame_ccc], float s_label,
                                  float s_dis[max_classss], const int hog_weight[max_classss],
                                  CvPoint cen_dis[frame_ccc], double z_value, double sum_distance, double sum_width,
                                  double sum_height, double seq_wid[frame_ccc], double seq_hei[frame_ccc],
                                  string pre_action) { // determining predict number of class
    //cout<<"determin_label4 IN: "<<frame_num<<endl;
    float last_label[max_class * 2], last_dis[max_class * 2];
    float detmer_dis[max_class];
    float max_label = -50, max_dist = -50, hog_max_dist = -50, stip_max_dist = -50;
    int str_cnt = 0, slp_cnt = 0;
    float sum_dis = 0;
    double shp_tra[frame_ccc][2];
    double sum_tra[2];
    for (int i = 0; i < frame_ccc; i++) {
        sum_dis += sqrt(pow(cen_dis[i].x, 2.0) + pow(cen_dis[i].y, 2.0));
    }
    if (z_value <= 2.3) {
        sum_dis = sum_dis * 2;
    }
    for (int i = 0; i < frame_ccc; i++) {
//		cout<<"[B] : {"<<cen_dis[i].x<<","<<cen_dis[i].y<<"}"<<endl;
        shp_tra[i][0] = cen_dis[i].x / sum_dis;
        sum_tra[0] += shp_tra[i][0];
        shp_tra[i][1] = cen_dis[i].y / sum_dis;
        sum_tra[1] += shp_tra[i][1];
//		cout<<"[A] : {"<<shp_tra[i][0]<<","<<shp_tra[i][1]<<"}"<<endl;
//		cout<<"[bal] seq_hei[i]/seq_wid[i] : "<<seq_hei[i]/seq_wid[i]<<endl;
        if (z_value < 2.3 && seq_hei[i] / seq_wid[i] >= 1.7) {
            str_cnt++;
        }
        else if (z_value >= 2.3 && seq_hei[i] / seq_wid[i] >= 1.5) {
            str_cnt++;
        }
        else if (seq_hei[i] / seq_wid[i] < 0.65)
//		else if(seq_hei[i]/seq_wid[i] < 1.2)
        {
            slp_cnt++;
        }
    }
    //cout << "[S1] : {" << sum_tra[0] << "," << sum_tra[1] << "}" << endl;
    //cout << "[S2] : {" << str_cnt << "," << slp_cnt << "}" << endl;
    for (int _j = 0; _j < max_class * 2; _j++) {
        if (_j < max_class) {
            float temp_l = 0, temp_d = 0;
            for (int _j2 = 0; _j2 < frame_con; _j2++) {
                temp_d += hog_dis[_j][_j2];
            }
            last_dis[_j] = temp_d / frame_con;
        }
        else {
            last_dis[_j] = s_dis[_j - max_class];
        }

    }
    if (slp_cnt < 4) {
        last_dis[2] -= 1;
    }
    if (str_cnt < 4) {
        last_dis[1] -= 1;
        last_dis[3] -= 1;
    }
    else {
        last_dis[1] += 0.7;
    }
    if (sum_width <= 100) {
        last_dis[4] -= 1;
    }
    if (sum_width >= 100 && sum_height < 180 && sum_distance < 130 && sum_tra[0] < 0.1) {
        //cout << "is handwaving?" << endl;
        last_dis[9 + max_class] += 1.85;
        last_dis[3 + max_class] -= 2;
    }

    if (sum_width >= 200 && sum_height < 180 && str_cnt >= 8) {
        //cout << "is kicking?" << endl;
        last_dis[7] += 1.45;
    }
    else {
        last_dis[7] -= 1.5;
    }
    if (sum_height > 100) {
        last_dis[3] = -10;
    }
// class-room db
    if (z_value <= 2.3) {
        last_dis[1] -= 1;
    }

    if (sum_height > 100) {
        last_dis[2] = -10;
    }
    if (slp_cnt < 4) {
        last_dis[2] -= 1;
    }
/*	if(sum_tra[1]>=0.36)
	{
		last_dis[0] += 2;
		last_dis[1] -= 2;
	}
	else if(sum_tra[1]<-0.3)
	{
		last_dis[1] += 2;
		last_dis[3] += 2;
	}
*/
    if ((sum_distance >= 100 && sum_distance < 330) && (sum_tra[1] >= 0.6) &&
        (sum_height >= 200 && sum_height <= 400)) {
        //cout << "is falling?" << endl;
        last_dis[5 + max_class] = (last_dis[5 + max_class] > 0 ? 1 : -1) * (abs(last_dis[5 + max_class]) + 1);
        last_dis[6 + max_class] = (last_dis[6 + max_class] > 0 ? 1 : -1) * (abs(last_dis[6 + max_class]) + 1);
        last_dis[8 + max_class] += 1.8;
    }
    else {
/*		last_dis[5+max_class] -= 1.8;
		last_dis[6+max_class] -= 1.8;
		last_dis[7] -= 1.8;
		last_dis[8+max_class] -= 1.8;
		last_dis[9+max_class] -= 0.8;*/
    }
//walk	together
    //	last_dis[5+max_class] += 1.8;

/*	if(abs(sum_tra[0])>=0.8){
		cout<<"is walking?"<<endl;
		last_dis[5+max_class] += 1.8;
		last_dis[6+max_class] += 1.4;
	}
*/
    //cout << "[S3] : {" << abs(sum_tra[0]) << "," << sum_height << "," << sum_width << "}" << endl;
/*	if((abs(sum_tra[0])<0.5)&&(sum_width>=310)&&(sum_height<120))
	{
		cout<<"add waving"<<endl;
		last_dis[9+max_class] += 1;
		last_dis[3+max_class] -= 1;
	}*/
    /*if(!moving)
	{
		cout<<"is no moving??"<<endl;
		last_dis[5+max_class] = (last_dis[5+max_class]>0?1:-1)*(abs(last_dis[5+max_class])+1);
		last_dis[6+max_class] = (last_dis[6+max_class]>0?1:-1)*(abs(last_dis[6+max_class])+1);
	}
	else
	{
		if((sum_distance >= 150 && sum_distance < 330) && ((sum_width >= 180 && sum_width <= 400) && (sum_height >= 200 && sum_height <= 400)))
		{
			cout<<"is falling?"<<endl;
			last_dis[5+max_class] = (last_dis[5+max_class]>0?1:-1)*(abs(last_dis[5+max_class])+1);
			last_dis[6+max_class] = (last_dis[6+max_class]>0?1:-1)*(abs(last_dis[6+max_class])+1);
			last_dis[8+max_class] += 0.8;
		}
		else
		{
			last_dis[6+max_class] -= 0.8;
			last_dis[8+max_class] -= 0.8;
		}
	}*/

    for (int _j = 0; _j < max_class; _j++) {

        double h_wei = ((double) hog_weight[_j] / (double) 10);

        double h_val = (double) ((h_wei) * (max_class + last_dis[_j]));
        double b_val = (double) (abs(1 - h_wei) * (max_class + last_dis[_j + max_class]));
        detmer_dis[_j] = h_val + b_val; //0.1~0.9 weight
        //cout << "determin_pair_label_i-" << _j << ", h_val : " << h_val << ", b_val : " << b_val
        //<< " , detmer_dis[_j] : " << detmer_dis[_j] << endl;
        if (max_dist < detmer_dis[_j]) {
            max_dist = detmer_dis[_j];
            max_label = _j + 1;
        }

    }
    //cout << "max_label : " << max_label << " , max_dist : " << max_dist << endl;
    if ((strcmp(pre_action.c_str(), "FALLING") == 0) && max_label == 3) {
        max_label = 9;
    }

    return max_label;
    //cout<<"determin_label4 OUT: "<<frame_num<<endl;
}

/*// violence sinario
int gr_module_4y::determin_label2(float hog_label[frame_ccc],float hog_dis[max_classss][frame_ccc],float s_label,float s_dis[max_classss],const int hog_weight[max_classss],CvPoint cen_dis[frame_ccc],double z_value,double sum_distance,double sum_width,double sum_height,double seq_wid[frame_ccc],double seq_hei[frame_ccc],string pre_action)
{ // determining predict number of class
	float last_label[max_class*2],last_dis[max_class*2];
	float detmer_dis[max_class];
	float max_label = -50,max_dist = -50,hog_max_dist = -50,stip_max_dist = -50;
	int str_cnt = 0,slp_cnt = 0;
	float sum_dis = 0;
	double shp_tra[frame_ccc][2];
	double sum_tra[2];
	for(int i = 0 ; i < frame_ccc;i++)
	{
		sum_dis += sqrt(pow(cen_dis[i].x,2.0)+pow(cen_dis[i].y,2.0));
	}
	if(z_value<=2.3)
	{
		sum_dis = sum_dis*2;
	}
	for(int i = 0 ; i < frame_ccc;i++)
	{
//		cout<<"[B] : {"<<cen_dis[i].x<<","<<cen_dis[i].y<<"}"<<endl;
		shp_tra[i][0] = cen_dis[i].x/sum_dis;
		sum_tra[0] += shp_tra[i][0];
		shp_tra[i][1] = cen_dis[i].y/sum_dis;
		sum_tra[1] += shp_tra[i][1];
//		cout<<"[A] : {"<<shp_tra[i][0]<<","<<shp_tra[i][1]<<"}"<<endl;
//		cout<<"[bal] seq_hei[i]/seq_wid[i] : "<<seq_hei[i]/seq_wid[i]<<endl;
		if(z_value<2.3 && seq_hei[i]/seq_wid[i] >= 1.7)
		{
			str_cnt++;
		}
		else if(z_value >= 2.3 && seq_hei[i]/seq_wid[i] >= 1.5)
		{
			str_cnt++;
		}
		else if(seq_hei[i]/seq_wid[i] < 0.65)
		{
			slp_cnt++;
		}
	}
	cout<<"[S1] : {"<<sum_tra[0]<<","<<sum_tra[1]<<"}"<<endl;
	cout<<"[S2] : {"<<str_cnt<<","<<slp_cnt<<"}"<<endl;
	for(int _j = 0 ; _j < max_class*2; _j++)	{
		if(_j<max_class)
		{
			float temp_l=0,temp_d=0;
			for(int _j2 = 0 ; _j2 < frame_con; _j2++)
			{
				temp_d += hog_dis[_j][_j2];
			}
			last_dis[_j] = temp_d/frame_con;
		}
		else
		{
			last_dis[_j] = s_dis[_j-max_class];
		}

	}
	if(slp_cnt < 4)
	{
		last_dis[2] -= 1;
	}
	if(str_cnt < 4)
	{
		last_dis[1] = -10;
		last_dis[3] = -10;
	}
	else
	{
		last_dis[1] += 1;
	}

	if(sum_width>=200 && sum_height<150 && str_cnt >= 8)
	{
		cout<<"is kicking?"<<endl;
		last_dis[7] += 1.65;
	}

	if(sum_height>100)
	{
		last_dis[3] = -10;
	}
	if((sum_distance >= 100 && sum_distance < 330) && (sum_tra[1]>=0.6) && (sum_height >= 200 && sum_height <= 400))
	{
		cout<<"is falling?"<<endl;
		last_dis[5+max_class] = (last_dis[5+max_class]>0?1:-1)*(abs(last_dis[5+max_class])+1);
		last_dis[6+max_class] = (last_dis[6+max_class]>0?1:-1)*(abs(last_dis[6+max_class])+1);
		last_dis[8+max_class] += 1.8;
	}
	else
	{
//		last_dis[5+max_class] -= 1.8;
//		last_dis[6+max_class] -= 1.8;
		last_dis[8+max_class] -= 0.8;
//		last_dis[9+max_class] -= 0.8;
	}
	if(abs(sum_tra[0])>=0.8){
		cout<<"is walking?"<<endl;
		last_dis[5+max_class] += 1.8;
		last_dis[6+max_class] += 1.4;
	}

	cout<<"[S3] : {"<<abs(sum_tra[0])<<","<<sum_height<<","<<sum_width<<"}"<<endl;

	for(int _j = 0 ; _j < max_class; _j++)	{

		double h_wei = ((double)hog_weight[_j]/(double)10);

		double h_val = (double)((h_wei)*(max_class+last_dis[_j]));
		double b_val = (double)(abs(1-h_wei)*(max_class+last_dis[_j+max_class]));
		detmer_dis[_j] = h_val + b_val; //0.1~0.9 weight
		cout<<"determin_pair_label_i-"<<_j<<", h_val : "<<h_val<<", b_val : "<<b_val<<" , detmer_dis[_j] : "<<detmer_dis[_j]<<endl;
		if(max_dist < detmer_dis[_j])
		{
			max_dist = detmer_dis[_j];
			max_label = _j+1;
		}

	}
	cout<<"max_label : "<<max_label<<" , max_dist : "<<max_dist<<endl;
	if((strcmp(pre_action.c_str(),"FALLING")==0)&&max_label==3)
	{
		max_label = 9;
	}

	return max_label;
}

int gr_module_4y::determin_hog_label(float hog_label[frame_ccc])
{
	int count_num[max_classss+1];
	for (int i = 0 ; i < max_classss+1 ; i++)
	{
		count_num[i] = 0;
	}
	for (int i = 0 ; i < frame_ccc; i++)
	{
		count_num[(int)hog_label[i]] = count_num[(int)hog_label[i]] + 1;

//		cout<<"determin_hog_label_i-"<<i<<" , hog_label[i] - "<<hog_label[i]<<", count_num[hog_label[i]] = "<<count_num[hog_label[i]]<<endl;
	}
	int max_num=-1,max_index=-1;
	for (int i = 0 ; i < max_classss+1 ; i++)
	{
		if (max_num < count_num[i])
		{
			max_num = count_num[i];
			max_index = i;
		}
	}
	return max_index;
}
int gr_module_4y::determin_hog_label(float hog_label[frame_ccc],bool moving)
{
	int count_num[max_classss+1];
	for (int i = 0 ; i < max_classss+1 ; i++)
	{
		count_num[i] = 0;
	}
	for (int i = 0 ; i < frame_ccc; i++)
	{
		count_num[(int)hog_label[i]] = count_num[(int)hog_label[i]] + 1;

//		cout<<"determin_hog_label_i-"<<i<<" , hog_label[i] - "<<hog_label[i]<<", count_num[hog_label[i]] = "<<count_num[(int)hog_label[i]]<<endl;
	}
	if(!moving)
	{
		count_num[5] = 0;
		count_num[6] = 0;
//		count_num[7] = 0;
		count_num[8] = 0;
	}
	int max_num=-1,max_index=-1;
	for (int i = 0 ; i < max_classss+1 ; i++)
	{
		if (max_num < count_num[i])
		{
			max_num = count_num[i];
			max_index = i;
		}
	}
	return max_index;
}

*/
int gr_module_4y::determin_hog_label(float hog_dis[max_classss][frame_ccc], bool moving) {
    //cout<<"determin_hog_label IN: "<<frame_num<<endl;
    float last_dis[max_classss], detmer_dis[max_classss];
    for (int _j = 0; _j < max_classss; _j++) {
        float temp_l = 0, temp_d = 0;
        for (int _j2 = 0; _j2 < frame_ccc; _j2++) {
            temp_d += hog_dis[_j][_j2];
            //		cout<<_j<<" : "<<_j2<<", hog_dis[_j][_j2] : "<<hog_dis[_j][_j2]<<endl;
        }
        last_dis[_j] = temp_d / frame_ccc;
        //	cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
    }
    if (!moving) {
        last_dis[5] = -50;
        last_dis[6] = -50;
        last_dis[8] = -50;
    }
    float max_dist = -50;
    int max_index = -50;
    for (int i = 0; i < max_classss; i++) {
        double h_wei = ((double) hog_weight[i] / (double) 10);
        double h_val = (double) (h_wei * (max_classss + last_dis[i] * 10));
        detmer_dis[i] = h_val;
        //	cout<<"determin_hog_label_i-"<<i<<", last_dis : "<<last_dis[i]<<" , detmer_dis[_j] : "<<detmer_dis[i]<<endl;
        if (max_dist < detmer_dis[i]) {
            max_dist = detmer_dis[i];

            max_index = i + 1;
        }
    }
//	cout<<"max_dist : " << max_dist <<endl;
    return max_index;
    //cout<<"determin_hog_label OUT: "<<frame_num<<endl;
}

int gr_module_4y::determin_hog_label2(float hog_label[frame_ccc], float hog_dis[max_classss][frame_ccc], bool moving) {
    //cout<<"determin_hog_label2 IN: "<<frame_num<<endl;
    int count_num[max_classss + 1];
    for (int i = 0; i < max_classss + 1; i++) {
        count_num[i] = 0;
    }
    for (int i = 0; i < frame_ccc; i++) {
        count_num[(int) hog_label[i]] = count_num[(int) hog_label[i]] + 1;
//		cout<<"determin_hog_label_i-"<<i<<" , hog_label[i] - "<<hog_label[i]<<", count_num[hog_label[i]] = "<<count_num[(int)hog_label[i]]<<endl;
    }
    float last_dis[max_classss], detmer_dis[max_classss];
    for (int _j = 0; _j < max_classss; _j++) {
        float temp_l = 0, temp_d = 0;
        for (int _j2 = 0; _j2 < frame_ccc; _j2++) {
            temp_d += hog_dis[_j][_j2];
            //		cout<<_j<<" : "<<_j2<<", hog_dis[_j][_j2] : "<<hog_dis[_j][_j2]<<endl;
        }
        last_dis[_j] = temp_d / frame_ccc;
        //	cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
    }
    if (!moving) {
        last_dis[5] = -50;
        last_dis[6] = -50;
        last_dis[8] = -50;
        last_dis[9] = -50;
    }
    float max_dist = -50;
    int max_index = -50;
    for (int i = 0; i < max_classss; i++) {
        double h_wei = ((double) hog_weight[i] / (double) 10);
        double h_val = (double) (h_wei * (max_classss + last_dis[i] * 5 + (count_num[i + 1] / frame_ccc) * 5));
        detmer_dis[i] = h_val;
        //	cout<<"determin_hog_label_i-"<<i<<", last_dis : "<<last_dis[i]<<" , detmer_dis[_j] : "<<detmer_dis[i]<<endl;
        if (max_dist < detmer_dis[i]) {
            max_dist = detmer_dis[i];

            max_index = i + 1;
        }
    }
//	cout<<"max_dist : " << max_dist <<endl;
    return max_index;
    //cout<<"determin_hog_label2 OUT: "<<frame_num<<endl;
}

int gr_module_4y::determin_hog_label3(float hog_label[frame_ccc], float hog_dis[max_classss][frame_ccc],
                                      double seq_wid[frame_ccc], double seq_hei[frame_ccc], double z_value) {
    //cout<<"determin_hog_label3 IN: "<<frame_num<<endl;
    int count_num[max_classss + 1];
    for (int i = 0; i < max_classss + 1; i++) {
        count_num[i] = 0;
    }
    int str_cnt = 0, slp_cnt = 0, ask_cnt = 0;
    for (int i = 0; i < frame_ccc; i++) {
        count_num[(int) hog_label[i]] = count_num[(int) hog_label[i]] + 1;
        //cout << seq_hei[i] / seq_wid[i] << endl;
        if (z_value < 2.3 && seq_hei[i] / seq_wid[i] >= 1.5) {
            str_cnt++;
        }
        else if (z_value >= 2.3 && seq_hei[i] / seq_wid[i] >= 1.1) {
            str_cnt++;
        }

        if (seq_hei[i] / seq_wid[i] < 0.7)
//		if(seq_hei[i]/seq_wid[i] < 1.2)
        {
            slp_cnt++;
        }
//		cout<<"determin_hog_label_i-"<<i<<" , hog_label[i] - "<<hog_label[i]<<", count_num[hog_label[i]] = "<<count_num[(int)hog_label[i]]<<endl;
    }
    float last_dis[max_classss], detmer_dis[max_classss];
    for (int _j = 0; _j < max_classss; _j++) {
        float temp_l = 0, temp_d = 0;
        for (int _j2 = 0; _j2 < frame_ccc; _j2++) {
            temp_d += hog_dis[_j][_j2];
            //		cout<<_j<<" : "<<_j2<<", hog_dis[_j][_j2] : "<<hog_dis[_j][_j2]<<endl;
        }
        last_dis[_j] = temp_d / frame_ccc;
        //	cout<<"temp_l : "<<temp_l<<", temp_d : "<<temp_d<<endl;
    }
    //cout << "str_cnt : " << str_cnt << ", slp_cnt : " << slp_cnt << endl;
    if (slp_cnt < 8) {
        last_dis[2] -= 2;
    }
    if (str_cnt < 8) {
        last_dis[1] = -50;
        last_dis[3] = -50;
    }
    else {
        last_dis[0] -= 0.5;
        last_dis[2] -= 0.5;
    }
    if (true) {
        last_dis[4] = -50;
        last_dis[5] = -50;
        last_dis[6] = -50;
        last_dis[7] = -50;
        last_dis[8] = -50;
        last_dis[9] = -50;
    }
    float max_dist = -50;
    int max_index = -50;
    for (int i = 0; i < max_classss; i++) {
        double h_wei = (((double) hog_weight[i] + 1) / (double) 10);
        //double h_val = (double)(h_wei*(max_classss+last_dis[i]+(count_num[i+1]/frame_ccc)*5));
        double h_val = (double) (1 * (max_classss + last_dis[i] + (count_num[i + 1] / frame_ccc) * 5));
        detmer_dis[i] = h_val;
        //cout << "determin_hog_label_i-" << i << ", last_dis : " << last_dis[i] << " , detmer_dis[_j] : "
        //<< detmer_dis[i] << endl;
        if (max_dist < detmer_dis[i]) {
            max_dist = detmer_dis[i];

            max_index = i + 1;
        }
    }
//	cout<<"max_dist : " << max_dist <<endl;
    return max_index;
    //cout<<"determin_hog_label3 OUT: "<<frame_num<<endl;
}

float gr_module_4y::pairwise_center_dis(CvPoint pre_p, CvPoint cur_p) {
    //cout<<"pairwise_center_dis IN: "<<frame_num<<endl;
    float dis;
    dis = pow(pre_p.x - cur_p.x, 2.0);
    dis = dis + pow(pre_p.y - cur_p.y, 2.0);
    dis = sqrt(dis);
    return dis;
    //cout<<"pairwise_center_dis OUT: "<<frame_num<<endl;
}

float gr_module_4y::pairwise_center_dis2(CvPoint pre_p, CvPoint cur_p) {
    //cout<<"pairwise_center_dis2 IN: "<<frame_num<<endl;
    float dis;
    dis = pow(pre_p.x - cur_p.x, 2.0);
    dis = dis + pow(pre_p.y - cur_p.y, 2.0) / 2;
    dis = sqrt(dis);
    return dis;
    //cout<<"pairwise_center_dis2 OUT: "<<frame_num<<endl;
}

float gr_module_4y::pairwise_width_dis(int pre_p, int cur_p) {
    //cout<<"pairwise_width_dis IN: "<<frame_num<<endl;
    float dis;
    dis = pow(pre_p - cur_p, 2.0);
//	dis = dis + pow(pre_p.y-cur_p.y,2.0)/2;
    dis = sqrt(dis);
    return dis;
    //cout<<"pairwise_width_dis OUT: "<<frame_num<<endl;
}

float gr_module_4y::pairwise_fall_dis(int pre_p, int cur_p) {
    //cout<<"pairwise_fall_dis IN: "<<frame_num<<endl;
    float dis;
    int temp = pre_p - cur_p;
    if (temp < 0)
        temp = 1;
    dis = pow(temp, 2.0);
//	dis = dis + pow(pre_p.y-cur_p.y,2.0)/2;
    dis = sqrt(dis);
    return dis;
    //cout<<"pairwise_fall_dis OUT: "<<frame_num<<endl;
}

float gr_module_4y::pairwise_fall_dis2(int pre_p, int cur_p, CvPoint pre_p2, CvPoint cur_p2) {
    //cout<<"pairwise_fall_dis2 IN: "<<frame_num<<endl;
    float dis;
    int temp = pre_p - cur_p;
    if (temp < 0)
        temp = 1;
    dis = pow(temp, 2.0);
//	dis = dis + pow(pre_p.y-cur_p.y,2.0)/2;
    dis = sqrt(dis);
//	cout<<"pre_p2.y - "<<pre_p2.y<<" , cur_p2.y - " <<cur_p2.y<<endl;
    float temp2 = cur_p2.y - pre_p2.y;
    if (temp2 < 0)
        temp2 = 0;

    return dis + temp2;
    //cout<<"pairwise_fall_dis2 OUT: "<<frame_num<<endl;
}

bool gr_module_4y::moving_jugment(double dis[frame_ccc]) {
    //cout<<"moving_jugment IN: "<<frame_num<<endl;
    double sum_dis = 0;
    for (int j = 0; j < frame_ccc; j++) {
        sum_dis = sum_dis + dis[j];
    }
//	cout<<"sum distance : "<<sum_dis<<endl;
    if (sum_dis >= 100) {
        return true;
    }
    return false;
    //cout<<"moving_jugment OUT: "<<frame_num<<endl;
}

bool gr_module_4y::moving_jugment(double dis[frame_ccc], double &dis_out) {
    //cout<<"moving_jugment2 IN: "<<frame_num<<endl;
    double sum_dis = 0;
    for (int j = 0; j < frame_con; j++) {
        sum_dis = sum_dis + dis[j];
    }
    //cout << "sum distance : " << sum_dis << endl;
    dis_out = sum_dis;
    if (sum_dis >= 100) {
        return true;
    }
    return false;
    //cout<<"moving_jugment2 OUT: "<<frame_num<<endl;
}

double gr_module_4y::sum_of_width_distance(double dis[frame_ccc]) {
    double sum_dis = 0;
    for (int j = 0; j < frame_con; j++) {
        sum_dis = sum_dis + dis[j];
    }
    return sum_dis;
}

void gr_module_4y::print_text(IplImage *frame_rgb, char *gro_result, CvPoint pt5, CvFont font, CvScalar color) {
    //cout<<"print_text IN: "<<frame_num<<endl;
    CvPoint temp1 = {pt5.x - 1, pt5.y};
    cvPutText(frame_rgb, gro_result, temp1, &font, CV_RGB(0, 0, 0));
    CvPoint temp2 = {pt5.x + 1, pt5.y};
    cvPutText(frame_rgb, gro_result, temp2, &font, CV_RGB(0, 0, 0));
    CvPoint temp3 = {pt5.x, pt5.y - 1};
    cvPutText(frame_rgb, gro_result, temp3, &font, CV_RGB(0, 0, 0));
    CvPoint temp4 = {pt5.x, pt5.y + 1};
    cvPutText(frame_rgb, gro_result, temp4, &font, CV_RGB(0, 0, 0));
    CvPoint temp5 = {pt5.x, pt5.y};
    cvPutText(frame_rgb, gro_result, temp5, &font, color);
    //cout<<"print_text OUT: "<<frame_num<<endl;
}

void gr_module_4y::print_text_d(IplImage *frame_rgb, char *gro_result, CvPoint pt5, CvFont font, CvScalar color) {
    //cout<<"print_text_d IN: "<<frame_num<<endl;
    CvPoint temp1 = {pt5.x - 1, pt5.y};
    cvPutText(frame_rgb, gro_result, temp1, &font, CV_RGB(255, 255, 255));
    CvPoint temp2 = {pt5.x + 1, pt5.y};
    cvPutText(frame_rgb, gro_result, temp2, &font, CV_RGB(255, 255, 255));
    CvPoint temp3 = {pt5.x, pt5.y - 1};
    cvPutText(frame_rgb, gro_result, temp3, &font, CV_RGB(255, 255, 255));
    CvPoint temp4 = {pt5.x, pt5.y + 1};
    cvPutText(frame_rgb, gro_result, temp4, &font, CV_RGB(255, 255, 255));
    CvPoint temp5 = {pt5.x, pt5.y};
    cvPutText(frame_rgb, gro_result, temp5, &font, color);
    //cout<<"print_text_d OUT: "<<frame_num<<endl;
}

void gr_module_4y::action_label_out(int ret, string &out) {
    //cout<<"action_label_out IN: "<<frame_num<<endl;

    switch (ret) {
        case 1:
            out = "SITTING";
            break;
        case 2:
            out = "STANDING";
            break;
        case 3:
            out = "SLEEPING";
            break;
        case 4:
            out = "ASKING";
            break;
        case 5:
            out = "BOXING";
            break;
        case 6:
            out = "WALKING";
            break;
        case 7:
            out = "RUNNING";
            break;
        case 8:
            out = "KICKING";
            break;
        case 9:
            out = "FALLING";
            break;
        case 10:
            out = "HAND_WAVING";
            break;
    }
    //cout<<"action_label_out OUT: "<<frame_num<<endl;
}

void
gr_module_4y::draw_fight(IplImage *frame_rgb, IplImage *frame_depth, int total, String stip_map_action[], CvFont font,
                         CvPoint pt1[], CvPoint pt2[], double z_value[]) {// finding fight sceen in group action
    //cout<<"draw_fight IN: "<<frame_num<<endl;
    bool iswarning = false;
    if (total >= 2) {
        bool up_rank = false;
        for (int i = 0; i < total; i++) {
            if (z_value[i] < 4) {
                draw_subject[i] = true;
            }
            else {
                draw_subject[i] = false;
            }
//			cout<<"["<<i<<"]"<<z_value[i]<<","<<strcmp(stip_map_action[i].c_str(),"BOXING")<<","<<stip_map_action[i].c_str()<<endl;
            if (strcmp(stip_map_action[i].c_str(), "BOXING") == 0 && draw_subject[i]) {
                fighter = i;
                //cout << "fighter??" << endl;
                for (int _i = 0; _i < total; _i++) {
                    if (_i != i && draw_subject[_i]) {
                        if (strcmp(stip_map_action[_i].c_str(), "BOXING") == 0) {
                            fighted = _i;
                            fight_acc += 2;
                            up_rank = true;
                            break;
                        }
                        if (strcmp(stip_map_action[_i].c_str(), "SLEEPING") == 0 ||
                            strcmp(stip_map_action[_i].c_str(), "SITTING") == 0) {
                            fighted = _i;
                            fight_acc++;
                            up_rank = true;
                            break;
                        }
                        if (strcmp(stip_map_action[_i].c_str(), "HAND_WAVING") == 0) {
                            fighted = _i;
                            fight_acc++;
                            iswarning = true;
                            up_rank = true;
                            break;
                        }
                    }
                }
            }
            if (up_rank) {
                break;
            }
        }
        if (!up_rank) {
            fight_acc--;
        }
    }
    else {
        fight_acc--;
    }
    if (fight_acc < 0) {
        fight_acc = 0;
    }
    if (fight_acc > 20) {
        fight_acc = 19;
    }
    //cout << "fight_acc : " << fight_acc << endl;
    if (fight_acc >= 15) {
        if (!iswarning) {
            if (total >= 2) {
                CvPoint t_rect1 = {MIN(pt1[fighter].x, pt1[fighted].x) - 40, MIN(pt1[fighter].y, pt1[fighted].y) - 40};
                CvPoint t_rect2 = {MAX(pt2[fighter].x, pt2[fighted].x) + 40, MAX(pt2[fighter].y, pt2[fighted].y) + 40};
                CvPoint t_rect3 = {MIN(pt1[fighter].x, pt1[fighted].x) - 40, MIN(pt1[fighter].y, pt1[fighted].y) - 50};
                cvRectangle(frame_rgb, t_rect1, t_rect2, CV_RGB(100, 0, 0), 1, 8, 0);
//				cvRectangle(frame_depth, t_rect1, t_rect2, CV_RGB(200,0,0), 1, 8, 0);

                print_text(frame_rgb, "Fight!!", t_rect3, font, CV_RGB(100, 0, 0));
//				print_text(frame_depth,"Fight!!",t_rect3,font,CV_RGB(200,0,0));
            }
        }
        else {
            if (total >= 2) {
//				CvPoint t_rect1 = {MIN(pt1[fighter].x,pt1[fighted].x)-40,MIN(pt1[fighter].y,pt1[fighted].y)-40};
//				CvPoint t_rect2 = {MAX(pt2[fighter].x,pt2[fighted].x)+40,MAX(pt2[fighter].y,pt2[fighted].y)+40};
//				CvPoint t_rect3 = {MIN(pt1[fighter].x,pt1[fighted].x)-40,MIN(pt1[fighter].y,pt1[fighted].y)-50};
                CvPoint t_rect1 = {5, 5};
                CvPoint t_rect2 = {640, 480};
                CvPoint t_rect3 = {20, 40};

                cvRectangle(frame_rgb, t_rect1, t_rect2, CV_RGB(255, 0, 0), 1, 8, 0);
                cvRectangle(frame_rgb, {pt1[fighted].x, pt1[fighted].y}, {pt2[fighted].x, pt2[fighted].y},
                            CV_RGB(255, 0, 0), 4, 8, 0);
//				cvRectangle(frame_depth, t_rect1, t_rect2, CV_RGB(255,0,0), 1, 8, 0);

                print_text(frame_rgb, "Warning!!Warning!!Warning!!", t_rect3, font, CV_RGB(255, 0, 0));
//				print_text(frame_depth,"Fight!!",t_rect3,font,CV_RGB(255,0,0));
            }
        }
    }
    //cout<<"draw_fight OUT: "<<frame_num<<endl;
}

void gr_module_4y::draw_violence(IplImage *frame_rgb, IplImage *frame_depth, int total, String stip_map_action[],
                                 CvFont font, CvPoint pt1[], CvPoint pt2[],
                                 double z_value[]) {// finding fight sceen in group action
    //cout<<"draw_violence IN: "<<frame_num<<endl;
    bool viowarning = false;
    int vio_sum = 0, global_vio_sum = 0;
    if (total >= 3) {
        bool up_rank = false;
        for (int i = 0; i < total; i++) {
            if (z_value[i] < 4) {
                draw_subject[i] = true;
            }
            else {
                draw_subject[i] = false;
            }
//			cout<<"["<<i<<"]"<<z_value[i]<<","<<strcmp(stip_map_action[i].c_str(),"BOXING")<<","<<stip_map_action[i].c_str()<<endl;
            if ((strcmp(stip_map_action[i].c_str(), "BOXING") == 0 ||
                 strcmp(stip_map_action[i].c_str(), "KICKING") == 0) && draw_subject[i]) {
//				fighter = i;
                vio_subject[i] = true;
                vio_sum++;
            }
            if (vio_subject[i] == true) {
                global_vio_sum++;
            }
        }
    }
    //cout << "vio_sum : " << vio_sum << "," << (double) ((vio_sum * 10 / total)) << endl;
    if ((double) ((global_vio_sum * 10 / total)) >= 6 && ((global_vio_sum >= 2 && vio_acc >= 1) || (vio_sum >= 2))) {
        bool up_rank = false;
        //cout << "Violence??" << endl;
        for (int _i = 0; _i < total; _i++) {
            if (vio_subject[_i] == false && draw_subject[_i]) {
                if (strcmp(stip_map_action[_i].c_str(), "SLEEPING") == 0 ||
                    strcmp(stip_map_action[_i].c_str(), "SITTING") == 0 ||
                    strcmp(stip_map_action[_i].c_str(), "STANDING") == 0) {
                    vio_ed = _i;
                    vio_acc += 1;
                    up_rank = true;
                    break;
                }
                if (strcmp(stip_map_action[_i].c_str(), "FALLING") == 0) {
                    //cout << "!!!!!!!!!!!!!!!!iswarning!!!!!!!!!!!!!!!" << endl;
                    vio_ed = _i;
                    vio_acc += 2;
                    viowarning = true;
                    up_rank = true;
                    break;
                }
            }
        }
        if (!up_rank) {
            vio_acc--;
        }
    }
    else {
        vio_acc--;
        global_vio_sum--;
    }
    if (vio_acc < 0) {
        vio_acc = 0;
    }
    if (vio_acc > 20) {
        vio_acc = 20;
    }
    if (global_vio_sum < 0) {
        global_vio_sum = 0;
        for (int _i = 0; _i < total; _i++) {
            vio_subject[_i] = false;
        }
    }
    if (global_vio_sum > 20) {
        global_vio_sum = 20;
    }
    //cout << "violence_acc : " << vio_acc << endl;
    if (vio_acc >= 7) {
        if (!viowarning) {
            if (total >= 3) {
                int pt1_x = 1000, pt1_y = 1000;
                int pt2_x = 0, pt2_y = 0;
                for (int _i = 0; _i < total; _i++) {
                    pt1_x = MIN(pt1[_i].x, pt1_x);
                    pt1_y = MIN(pt1[_i].y, pt1_y);
                    pt2_x = MAX(pt2[_i].x, pt2_x);
                    pt2_y = MAX(pt2[_i].y, pt2_y);
                }
                CvPoint t_rect1 = {pt1_x - 40, pt1_y - 40};
                CvPoint t_rect2 = {pt2_x + 40, pt2_y + 40};
                CvPoint t_rect3 = {pt1_x - 40, pt1_y - 50};
                cvRectangle(frame_rgb, t_rect1, t_rect2, CV_RGB(200, 200, 0), 1, 8, 0);
//				cvRectangle(frame_depth, t_rect1, t_rect2, CV_RGB(200,0,0), 1, 8, 0);

                print_text(frame_rgb, "Violence!!", t_rect3, font, CV_RGB(200, 200, 0));
//				print_text(frame_depth,"Fight!!",t_rect3,font,CV_RGB(200,0,0));
            }
        }
        else {
            if (total >= 3) {

            }
            else {
                vio_acc--;

            }
            CvPoint t_rect1 = {5, 5};
            CvPoint t_rect2 = {640, 480};
            CvPoint t_rect3 = {20, 40};

            cvRectangle(frame_rgb, t_rect1, t_rect2, CV_RGB(255, 0, 0), 1, 8, 0);
            cvRectangle(frame_rgb, {pt1[vio_ed].x, pt1[vio_ed].y}, {pt2[vio_ed].x, pt2[vio_ed].y}, CV_RGB(255, 0, 0), 4,
                        8, 0);
            print_text(frame_rgb, "Warning!!Warning!!Warning!!", t_rect3, font, CV_RGB(255, 0, 0));
        }
    }
    else {
        viowarning = false;
    }
    //cout<<"draw_violence OUT: "<<frame_num<<endl;
}

void gr_module_4y::draw_walk_toge(IplImage *frame_rgb, IplImage *frame_depth, int total, String pre_action[],
                                  String stip_map_action[], CvFont font, CvPoint pt1[], CvPoint pt2[],
                                  double z_value[]) {// finding fight sceen in group action
    //cout<<"draw_walk_toge IN: "<<frame_num<<endl;
    bool viowarning = false;
    int walk_sum = 0;
    if (total >= 2) {
        bool up_rank = false;
        for (int i = 0; i < total; i++) {
            if (z_value[i] < 4) {
                draw_subject[i] = true;
            }
            else {
                draw_subject[i] = false;
            }
//			cout<<"["<<i<<"]"<<z_value[i]<<","<<strcmp(stip_map_action[i].c_str(),"BOXING")<<","<<stip_map_action[i].c_str()<<endl;
            if ((strcmp(stip_map_action[i].c_str(), "WALKING") == 0 ||
                 strcmp(stip_map_action[i].c_str(), "RUNNING") == 0) && draw_subject[i]) {
//				fighter = i;
                walk_subject[i] = true;
                walk_sum++;
                global_walk_sum++;
            }
            else {
                global_walk_sum--;
            }
        }
    }
    //cout << "walk_sum : " << walk_sum << "," << "global_walk_sum : " << global_walk_sum << ","
    //<< (double) ((walk_sum * 10) / total) << endl;
    if ((global_walk_sum >= 2 && walk_sum >= 2) || walk_sum >= 3 || global_walk_sum >= 3) {
//		cout<<"a"<<endl;
        int pt1_x = 1000, pt1_y = 1000;
        int pt2_x = 0, pt2_y = 0;

        for (int _i = 0; _i < total; _i++) {
            pt1_x = MIN(pt1[_i].x, pt1_x);
            pt1_y = MIN(pt1[_i].y, pt1_y);
            pt2_x = MAX(pt2[_i].x, pt2_x);
            pt2_y = MAX(pt2[_i].y, pt2_y);
        }

        CvPoint t_rect0 = {20, 40};
        CvPoint t_rect1 = {5, pt1_y - 40};
        CvPoint t_rect2 = {640, pt2_y + 40};
        CvPoint t_rect3 = {30, pt2_y - 20};
        cvRectangle(frame_rgb, t_rect1, t_rect2, CV_RGB(100, 100, 100), 4, 8, 0);
//				cvRectangle(frame_depth, t_rect1, t_rect2, CV_RGB(200,0,0), 1, 8, 0);

        print_text(frame_rgb, "Walk Path", t_rect3, font, CV_RGB(100, 100, 100));
        print_text(frame_rgb, "Walk Together", t_rect0, font, CV_RGB(0, 0, 200));
//				print_text(frame_depth,"Fight!!",t_rect3,font,CV_RGB(200,0,0));

    }
    //cout<<"draw_walk_toge OUT: "<<frame_num<<endl;
}

void gr_module_4y::gather_jud(IplImage *frame_rgb, IplImage *frame_depth, Mat depth_frame, int total,
                              String stip_map_action[], CvFont font, int pt1x_m[max_peoples], int pt1y_m[max_peoples],
                              int pt2x_m[max_peoples], int pt2y_m[max_peoples], bool isgather) {
    //cout<<"gather_jud IN: "<<frame_num<<endl;
    bool gather_peo[max_peoples];

    int thres_hold_meter = 240;
    int thres_hold_depth = 80;
    int true_val = 0;
    if (total >= 3) {
        for (int i = 0; i < total; i++) {
            for (int j = i + 1; j < total; j++) {
                int dist = sqrt(pow(pt1x_m[i] - pt2x_m[j], 2) + pow(pt1y_m[i] - pt2y_m[j], 2));

                int depth_dist = sqrt(pow((((int) (depth_frame.at<float>(pt1y_m[i], pt1x_m[i]))) * 100 -
                                           ((int) (depth_frame.at<float>(pt2y_m[i], pt2x_m[i]))) * 100), 2));
                //cout << dist << "," << depth_dist << endl;
                if (dist <= thres_hold_meter && depth_dist <= thres_hold_depth) {
                    gather_peo[i] = true;
                    gather_peo[j] = true;
                    true_val++;
                }
            }
        }
    }
    else {
        isgather = false;
    }
    if (true_val >= 2) {
        CvPoint t_rect = {20, 20};
        print_text(frame_rgb, "Gather : ", t_rect, font, CV_RGB(0, 0, 255));
        print_text(frame_depth, "Gather : ", t_rect, font, CV_RGB(0, 0, 255));
        int cmt = 0;
        for (int i = 0; i < total; i++) {
            if (gather_peo[i]) {
                CvPoint t_rect1 = {130 + cmt * 40, 20};
                char ouput_text[30];
                snprintf(ouput_text, 30, "%d ,", i);
                print_text(frame_rgb, ouput_text, t_rect1, font, CV_RGB(0, 0, 255));
                print_text(frame_depth, ouput_text, t_rect1, font, CV_RGB(0, 0, 255));
                cmt++;
            }
        }
        if (cmt == 0) {
            isgather = false;
        }
        else {
            isgather = true;;
        }
    }
    else {
        isgather = false;
    }
    //cout<<"gather_jud OUT: "<<frame_num<<endl;
}

void gr_module_4y::recognization(String *output_label, int isoutput[]) {
//    clock_t begin, end;
//    begin = clock();
    //cout<<"hog extraction part"<<endl;
    //cout<<"recognization IN: "<<frame_num<<endl;

    // ================================================================================
    // Classify each human object's appearance using HOG descriptor
    // ================================================================================
    for (int i = 0; i < _total; i++) {
//		cout<<"part 0,"<<_total<<t_image.size()<<endl;
//		cout<<pt1[i].x<<","<<pt1[i].y<<","<<pt2[i].x<<","<<pt2[i].y<<endl;
        cv::Mat piece0 = t_image(cv::Rect(pt1[i], pt2[i]));

        //cout<<"part A"<<endl;

        // ================================================================================
        // Extract HOG descriptor with spatial pyramid
        // ================================================================================
        Mat piece1, piece2, piece3, piece4;
        vector<float> descriptors0, descriptors1, descriptors2, descriptors3, descriptors4;

        // Construct the spatial pyramid of the input image
        piece1 = piece0(cv::Rect(0, 0, (int) (piece0.size().width / 2), (int) (piece0.size().height / 2)));
        piece2 = piece0(cv::Rect(0, (int) (piece0.size().height / 2), (int) (piece0.size().width / 2),
                                 (int) (piece0.size().height / 2)));
        piece3 = piece0(cv::Rect((int) (piece0.size().width / 2), 0, (int) (piece0.size().width / 2),
                                 (int) (piece0.size().height / 2)));
        piece4 = piece0(cv::Rect((int) (piece0.size().width / 2), (int) (piece0.size().height / 2),
                                 (int) (piece0.size().width / 2), (int) (piece0.size().height / 2)));
//		cout<<"part B"<<endl;
        cv::resize(piece0, piece0, cv::Size(64, 64), 0, 0);
        cv::resize(piece1, piece1, cv::Size(64, 64), 0, 0);
        cv::resize(piece2, piece2, cv::Size(64, 64), 0, 0);
        cv::resize(piece3, piece3, cv::Size(64, 64), 0, 0);
        cv::resize(piece4, piece4, cv::Size(64, 64), 0, 0);

        // Get HOG descriptors
        roi_hog->compute(piece0, descriptors0, Size(1, 1), Size(0, 0));
        roi_hog->compute(piece1, descriptors1, Size(1, 1), Size(0, 0));
        roi_hog->compute(piece2, descriptors2, Size(1, 1), Size(0, 0));
        roi_hog->compute(piece3, descriptors3, Size(1, 1), Size(0, 0));
        roi_hog->compute(piece4, descriptors4, Size(1, 1), Size(0, 0));
//		cout<<"part C"<<endl;

        // Concatenate descriptors
        descriptors0.reserve(descriptors0.size() * 5);
        descriptors0.insert(descriptors0.end(), descriptors1.begin(), descriptors1.end());
        descriptors0.insert(descriptors0.end(), descriptors2.begin(), descriptors2.end());
        descriptors0.insert(descriptors0.end(), descriptors3.begin(), descriptors3.end());
        descriptors0.insert(descriptors0.end(), descriptors4.begin(), descriptors4.end());
        Mat hog_descriptor = Mat(descriptors0, true);
        // ================================================================================

//		cout<<"part D"<<endl;

        // ================================================================================
        // Classify the HOG descriptor
        // ================================================================================
        int result = -1;
        result = do_hri_pose_predict(hog_descriptor,pose_dis)+1;
//		cout<<"part E"<<endl;

        // Do some heuristic process
        for (int _j = 0; _j < max_classss; _j++) {
            if (_j == 5 || _j == 6)
                continue;
            if (_j >= 5) {
                hog_dis[i][_j][circle_hog[i]] = pose_dis[_j - 2];
            }
            else {
                hog_dis[i][_j][circle_hog[i]] = pose_dis[_j];
            }

        }
        hog_label[i][circle_hog[i]++] = result;
        action_label_out(result, detcom_map_action2[i]);
        // ================================================================================

//		cout<<"part F"<<endl;
        //	t_image.release();
        piece0.release();
        piece1.release();
        piece2.release();
        piece3.release();
        piece4.release();

        descriptors0.clear();
        descriptors1.clear();
        descriptors2.clear();
        descriptors3.clear();
        descriptors4.clear();
//		cout<<"part G"<<endl;
    } // for (int i = 0; i < _total; i++)
    // ================================================================================

//	cout<<"DT extraction part"<<endl;
    // ================================================================================
    // Classify each human object's motion using DETCOM descriptor
    // ================================================================================
    if (frame_num == 0) { // initialization
        //      cout<<"dense initialization"<<endl;
        image_frame.create(frame_rgb.size(), CV_8UC3);
        //grey.create(frame_rgb.size(), CV_8UC1);
        prev_grey.create(frame_rgb.size(), CV_8UC1);
        //     cout<<"point 1"<<endl;
        InitPry(frame_rgb, fscales, sizes);//프레임의 scale-pyramid 생성

        BuildPry(sizes, CV_8UC1, prev_grey_pyr);
        BuildPry(sizes, CV_8UC1, grey_pyr);

        BuildPry(sizes, CV_32FC2, flow_pyr);
        BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
        BuildPry(sizes, CV_32FC(5), poly_pyr);
        //     cout<<"point 2"<<endl;
        xyScaleTracks.resize(scale_num);

        frame_rgb.copyTo(image_frame);
        cvtColor(image_frame, prev_grey, CV_BGR2GRAY);
        //    cout<<"point 3"<<endl;
        for (int iScale = 0; iScale < scale_num; iScale++) {
            if (iScale == 0)
                prev_grey.copyTo(prev_grey_pyr[0]);
            else
                resize(prev_grey_pyr[iScale - 1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0,
                       INTER_LINEAR);

            // dense sampling feature points
            std::vector <Point2f> points(0);
            DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

            // save the feature points
            std::list <Track> &tracks = xyScaleTracks[iScale];
            for (int i = 0; i < points.size(); i++)
                tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
        }
        //     cout<<"point 4"<<endl;
        // compute polynomial expansion
        my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

        frame_num++;
        //    cout<<"point 5"<<endl;
        //continue;
    }
    else {
        // 	cout<<"countiue part"<<endl;
        init_counter++;
        //    str2 = clock();
        frame_rgb.copyTo(image_frame);
        cvtColor(image_frame, grey, CV_BGR2GRAY);
        //      str3 = clock();
        //      cout<<"C 1"<<endl;
        // compute optical flow for all scales once
        my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
        my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);
        //      end3 = clock();
//		printf("poly Time : %f\n", ((double)(end3-str3)) / CLOCKS_PER_SEC);
        //       cout<<"C 2"<<endl;

        //================================================================================
        // Extract DT over the whole frame
        //================================================================================
        for (int iScale = 0; iScale < scale_num; iScale++) {
//			cout<<"a1"<<endl;
            if (iScale == 0)
                grey.copyTo(grey_pyr[0]);
            else
                resize(grey_pyr[iScale - 1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
//			cout<<"a2"<<endl;
            int width = grey_pyr[iScale].cols;
            int height = grey_pyr[iScale].rows;
//            str1 = clock();
            // compute the integral histograms
//			cout<<"a3"<<endl;
            DescMat *hogMat = InitDescMat(height + 1, width + 1, hogInfo.nBins);
//			cout<<"a4"<<endl;
            HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);
//			cout<<"a5"<<endl;
//            end1 = clock();
//			printf("1-th Time : %f\n", ((double)(end1-str1)) / CLOCKS_PER_SEC);
//            str1 = clock();
            DescMat *hofMat = InitDescMat(height + 1, width + 1, hofInfo.nBins);
            HofComp(flow_pyr[iScale], hofMat->desc, hofInfo);
//            end1 = clock();
//			printf("2-th Time : %f\n", ((double)(end1-str1)) / CLOCKS_PER_SEC);
//            str1 = clock();
            DescMat *mbhMatX = InitDescMat(height + 1, width + 1, mbhInfo.nBins);
            DescMat *mbhMatY = InitDescMat(height + 1, width + 1, mbhInfo.nBins);
            MbhComp(flow_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);
//            end1 = clock();
//			printf("3-th Time : %f\n", ((double)(end1-str1)) / CLOCKS_PER_SEC);
            //          cout<<"C 3"<<endl;

            // track feature points in each scale separately
            std::list <Track> &tracks = xyScaleTracks[iScale];
            for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
                int index = iTrack->index;
                Point2f prev_point = iTrack->point[index];
                int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width - 1);
                int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height - 1);

                Point2f point;
                point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2 * x];
                point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2 * x + 1];

                if (point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
                    iTrack = tracks.erase(iTrack);
                    continue;
                }

                // get the descriptors for the feature point
                RectInfo rect;
                GetRect(prev_point, rect, width, height, hogInfo);
                GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
                GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
                GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
                GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
                //	GetHOPC(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
                iTrack->addPoint(point);

                // draw the trajectories at the first scale
                if (show_track == 1 && iScale == 0)
                    DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image_frame);

                // if the trajectory achieves the maximal length
                if (iTrack->index >= trackInfo.length) {
                    std::vector <Point2f> trajectory(trackInfo.length + 1);
                    for (int i = 0; i <= trackInfo.length; ++i) {
                        trajectory[i] = iTrack->point[i] * fscales[iScale];
                        //cout<<"( "<<trajectory[i].x<<" , "<<trajectory[i].y<<" ) "<<endl;
                    }

                    float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
                    if (IsValid(trajectory, mean_x, mean_y, var_x, var_y, length)) { // 트레젝토리에서 정보 생성
                        if (test_bool) {
                            //cout<<"test code"<<endl;
                            for (int i = 0; i < _total; i++) {
                                CvPoint min_lt = min_roi(roi_h_lt[i], i);
                                CvPoint max_rb = max_roi(roi_h_rb[i], i);
                                // cout<<"["<<min_lt.x<<","<<min_lt.y<<"]["<<max_rb.x<<","<<max_rb.y<<"]"<<endl;
                                //cout<<"inroi : "<<inroi(min_lt,max_rb,mean_x,mean_y)<<endl;
                                if (inroi(min_lt, max_rb, mean_x, mean_y)) {
                                    if (first_f_bool[i]) {
                                        //	cout<<"First save"<<endl;
                                        SaveDesc(iTrack->hog, hogInfo, trackInfo, first_hog_storage[i]);
                                        SaveDesc(iTrack->hof, hofInfo, trackInfo, first_hof_storage[i]);
                                        SaveDesc(iTrack->mbhX, mbhInfo, trackInfo, first_mbhx_storage[i]);
                                        SaveDesc(iTrack->mbhY, mbhInfo, trackInfo, first_mbhy_storage[i]);
                                        first_f_cont[i]++;
                                    }
                                    if (second_f_bool[i]) {
                                        //	cout<<"Second save"<<endl;
                                        SaveDesc(iTrack->hog, hogInfo, trackInfo, second_hog_storage[i]);
                                        SaveDesc(iTrack->hof, hofInfo, trackInfo, second_hof_storage[i]);
                                        SaveDesc(iTrack->mbhX, mbhInfo, trackInfo, second_mbhx_storage[i]);
                                        SaveDesc(iTrack->mbhY, mbhInfo, trackInfo, second_mbhy_storage[i]);
                                        second_f_cont[i]++;
                                    }
                                }
                            }
                        }

                        //
                        if (print_bool) {
                            printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t\n", frame_num - f_length, frame_num, mean_x,
                                   mean_y, var_x, var_y, length, fscales[iScale]);
//							printf("%f\t%f\t%f\t\n",std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999),std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999),std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));

                            for (int i = 0; i < trackInfo.length; ++i) {
                                printf("%f\t%f\t", iTrack->point[i].x, iTrack->point[i].y);
                            }
                            printf("\n");
                            for (int i = 0; i < trackInfo.length; ++i) {
                                printf("%f\t%f\t", trajectory[i].x, trajectory[i].y);
                            }
                            printf("\n");
                            PrintDesc(iTrack->hog, hogInfo, trackInfo);
                            printf("\n");
                            PrintDesc(iTrack->hof, hofInfo, trackInfo);
                            printf("\n");
                            PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
                            printf("\n");
                            PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
                            printf("\n");
                        }
                        if (wirte_bool) {/*
							FILE * fp2 = fopen(feature_name, "at");
							fprintf(fp2, "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t",(float)(frame_num-f_length+1), (float)(frame_num+1), mean_x, mean_y, var_x, var_y, length, fscales[iScale]);
							fprintf(fp2, "%f\t%f\t%f\t\n",std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999),std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999),std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
							for (int i = 0; i < trackInfo.length; ++i)
							{
								fprintf(fp2, "%f\t%f\t",iTrack->point[i].x,iTrack->point[i].y);
							}
							fprintf(fp2,"\n");
							for (int i = 0; i < trackInfo.length; ++i)
							{
								fprintf(fp2, "%f\t%f\t\n",trajectory[i].x,trajectory[i].y);
							}
							fprintf(fp2,"\n");
							WriteDesc(iTrack->hog, hogInfo, trackInfo,fp2);
							fprintf(fp2,"\n");
							WriteDesc(iTrack->hof, hofInfo, trackInfo,fp2);
							fprintf(fp2,"\n");
							WriteDesc(iTrack->mbhX, mbhInfo, trackInfo,fp2);
							fprintf(fp2,"\n");
							WriteDesc(iTrack->mbhY, mbhInfo, trackInfo,fp2);
							fprintf(fp2, "\n\n");
							fclose(fp2);*/
                        }
                    }

                    iTrack = tracks.erase(iTrack);
                    continue;
                }
                ++iTrack;
            }
            ReleDescMat(hogMat);
            ReleDescMat(hofMat);
            ReleDescMat(mbhMatX);
            ReleDescMat(mbhMatY);

            if (init_counter != trackInfo.gap)
                continue;

            // detect new feature points every initGap frames
            std::vector <Point2f> points(0);
            for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
                points.push_back(iTrack->point[iTrack->index]);

            DenseSample(grey_pyr[iScale], points, quality, min_distance);
            // save the new feature points
            for (int i = 0; i < points.size(); i++)
                tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
        } // for (int iScale = 0; iScale < scale_num; iScale++)
        //================================================================================
//        cout<<"C 5"<<endl;

        //================================================================================
        // Classify each human object's appearance using DETCOM descriptor
        //================================================================================
        for (int i = 0; i < _total; i++) {
            //	cout<<"["<<i<<"] input "<<endl;
            move_dis[i][circle_hog[i] - 1] = pairwise_center_dis2(pre_cen_p[i], cur_cen_p[i]);
            move_dis2[i][circle_hog[i] - 1] = pairwise_width_dis(pre_wid_p[i], cur_wid_p[i]);
            move_dis3[i][circle_hog[i] - 1] = pairwise_fall_dis2(pre_hei_p[i], cur_hei_p[i], pre_cen_p[i],
                                                                 cur_cen_p[i]);
            seq_wid[i][circle_hog[i] - 1] = cur_wid_p[i];
            seq_hei[i][circle_hog[i] - 1] = cur_hei_p[i];
            cen_dis[i][circle_hog[i] - 1] = {cur_cen_p[i].x - pre_cen_p[i].x, cur_cen_p[i].y - pre_cen_p[i].y};
//			cout<<"["<<i<<"] dis : {"<<cen_dis[i][circle_hog[i]-1].x<<","<<cen_dis[i][circle_hog[i]-1].y<<"}"<<endl;
            pre_cen_p[i] = cur_cen_p[i];
            pre_wid_p[i] = cur_wid_p[i];
            pre_hei_p[i] = cur_hei_p[i];


            // First sliding window
            if (first_f_bool[i]) {
                //cout<<"FIRST SLIDING WINDOW START : "<<frame_num<<"\n"<<endl;
                if (first_f_num[i] >= frame_ccc) { // for each frame_ccc'th frame length
                    first_f_bool[i] = false;
                    second_f_bool[i] = true;
                    first_f_num[i] = 0;
                    //cout<<first_hog_storage[i].size()<<endl;
                    //              cout<<"first_f_cont[i] : "<<first_f_cont[i]<<endl;
                    Mat dest_hog, dest_hof, dest_mbhx, dest_mbhy;
                    double sum_dis = 0;
                    moving_jug[i] = moving_jugment(move_dis[i], sum_dis);
                    double sum_width = sum_of_width_distance(move_dis2[i]);
                    //cout << "[" << i << "] sum_width : " << sum_width << endl;
                    double sum_height = sum_of_width_distance(move_dis3[i]);
                    //cout << "[" << i << "] sum_height : " << sum_height << endl;

                    if (first_f_cont[i] < hofInfo.dim * 3) {
                        //                  	cout<<"no motion"<<endl;
                        no_motion[i] = true;
                        //  continue;
                    }
                    else if (first_f_cont[i] < hofInfo.dim * 3) {
                        //                      cout<<"under_output"<<endl;
                        no_motion[i] = false;
                        std::vector<float> ext_hog(first_hog_storage[i]);
                        ext_hog.reserve(first_hog_storage[i].size() * 3);
                        ext_hog.insert(ext_hog.end(), first_hog_storage[i].begin(), first_hog_storage[i].end());
                        ext_hog.insert(ext_hog.end(), first_hog_storage[i].begin(), first_hog_storage[i].end());
                        std::vector<float> ext_hof(first_hof_storage[i]);
                        ext_hof.reserve(first_hof_storage[i].size() * 3);
                        ext_hof.insert(ext_hof.end(), first_hof_storage[i].begin(), first_hof_storage[i].end());
                        ext_hof.insert(ext_hof.end(), first_hof_storage[i].begin(), first_hof_storage[i].end());
                        std::vector<float> ext_mbhx(first_mbhx_storage[i]);
                        ext_mbhx.reserve(first_mbhx_storage[i].size() * 3);
                        ext_mbhx.insert(ext_mbhx.end(), first_mbhx_storage[i].begin(), first_mbhx_storage[i].end());
                        ext_mbhx.insert(ext_mbhx.end(), first_mbhx_storage[i].begin(), first_mbhx_storage[i].end());
                        std::vector<float> ext_mbhy(first_mbhy_storage[i]);
                        ext_mbhy.reserve(first_mbhy_storage[i].size() * 3);
                        ext_mbhy.insert(ext_mbhy.end(), first_mbhy_storage[i].begin(), first_mbhy_storage[i].end());
                        ext_mbhy.insert(ext_mbhy.end(), first_mbhy_storage[i].begin(), first_mbhy_storage[i].end());

                        Mat cvt_hog(ext_hog, true);
                        Mat cvt_hof(ext_hof, true);
                        Mat cvt_mbhx(ext_mbhx, true);
                        Mat cvt_mbhy(ext_mbhy, true);

                        Mat data_hog = Mat::zeros(hogInfo.dim * 3, first_f_cont[i], CV_32FC1);
                        Mat data_mbhx = Mat::zeros(mbhInfo.dim * 3, first_f_cont[i], CV_32FC1);
                        Mat data_mbhy = Mat::zeros(mbhInfo.dim * 3, first_f_cont[i], CV_32FC1);
                        Mat data_hof = Mat::zeros(hofInfo.dim * 3, first_f_cont[i], CV_32FC1);

                        int temp_hog = 0, temp_hof = 0;
                        for (int ind_w = 0; ind_w < first_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hogInfo.dim * 3; ind_h++) {
                                data_hog.at<float>(ind_h, ind_w) = cvt_hog.at<float>(temp_hog, 0);
                                data_mbhx.at<float>(ind_h, ind_w) = cvt_mbhx.at<float>(temp_hog, 0);
                                data_mbhy.at<float>(ind_h, ind_w) = cvt_mbhy.at<float>(temp_hog++, 0);
                            }
                        }
                        for (int ind_w = 0; ind_w < first_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hofInfo.dim * 3; ind_h++) {
                                data_hof.at<float>(ind_h, ind_w) = cvt_hof.at<float>(temp_hof++, 0);
                            }
                        }

                        dest_hog = Mat(data_hog).clone().t();
                        first_hog_storage[i].clear();

                        dest_hof = Mat(data_hof).clone().t();
                        first_hof_storage[i].clear();

                        dest_mbhx = Mat(data_mbhx).clone().t();
                        first_mbhx_storage[i].clear();

                        dest_mbhy = Mat(data_mbhy).clone().t();
                        first_mbhy_storage[i].clear();

                        cvt_hog.release();
                        cvt_hof.release();
                        cvt_mbhx.release();
                        cvt_mbhy.release();
                        data_hog.release();
                        data_hof.release();
                        data_mbhx.release();
                        data_mbhy.release();
                    }
                    else {
                        //                      cout<<"normal output"<<endl;
                        no_motion[i] = false;
                        Mat cvt_hog(first_hog_storage[i], true);
                        Mat cvt_hof(first_hof_storage[i], true);
                        Mat cvt_mbhx(first_mbhx_storage[i], true);
                        Mat cvt_mbhy(first_mbhy_storage[i], true);

                        Mat data_hog = Mat::zeros(hogInfo.dim * 3, first_f_cont[i], CV_32FC1);
                        Mat data_mbhx = Mat::zeros(mbhInfo.dim * 3, first_f_cont[i], CV_32FC1);
                        Mat data_mbhy = Mat::zeros(mbhInfo.dim * 3, first_f_cont[i], CV_32FC1);
                        Mat data_hof = Mat::zeros(hofInfo.dim * 3, first_f_cont[i], CV_32FC1);

                        int temp_hog = 0, temp_hof = 0;
                        for (int ind_w = 0; ind_w < first_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hogInfo.dim * 3; ind_h++) {
                                data_hog.at<float>(ind_h, ind_w) = cvt_hog.at<float>(temp_hog, 0);
                                data_mbhx.at<float>(ind_h, ind_w) = cvt_mbhx.at<float>(temp_hog, 0);
                                data_mbhy.at<float>(ind_h, ind_w) = cvt_mbhy.at<float>(temp_hog++, 0);
                            }
                        }
                        for (int ind_w = 0; ind_w < first_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hofInfo.dim * 3; ind_h++) {
                                data_hof.at<float>(ind_h, ind_w) = cvt_hof.at<float>(temp_hof++, 0);
                            }
                        }

                        dest_hog = Mat(data_hog).clone().t();
                        first_hog_storage[i].clear();

                        dest_hof = Mat(data_hof).clone().t();
                        first_hof_storage[i].clear();

                        dest_mbhx = Mat(data_mbhx).clone().t();
                        first_mbhx_storage[i].clear();

                        dest_mbhy = Mat(data_mbhy).clone().t();
                        first_mbhy_storage[i].clear();

                        cvt_hog.release();
                        cvt_hof.release();
                        cvt_mbhx.release();
                        cvt_mbhy.release();
                        data_hog.release();
                        data_hof.release();
                        data_mbhx.release();
                        data_mbhy.release();
                    }
                    //         cout<<"part a"<<endl;
                    //         no_motion[i] = true;
                    if (no_motion[i]) {
//						cout<<"["<<i<<"] no_motion output!!!"<<endl;
                        float result = -1;
                        //			result = determin_hog_label2(hog_label[i],hog_dis[i],moving_jug[i]);
                        result = determin_hog_label3(hog_label[i], hog_dis[i], seq_wid[i], seq_hei[i], id_z_value[i]);

                        action_label_out(result, detcom_map_action[i]);
                        isoutput[i] = 1;
                        output_label[i] = detcom_map_action[i];
                        //			cout<<"resutl : "<<detcom_map_action[i]<<endl;
                    }
                    else {
//						cout<<"{"<<i<<"} Motion output"<<endl;
                        Mat dest_all = Mat::zeros(hog_factorial + hof_factorial + mbh_factorial + mbh_factorial, 1,
                                                  CV_32FC1);
                        CvMat *output_hog = cvCreateMat(hog_factorial, 1, CV_32FC1);
                        CvMat *output_hof = cvCreateMat(hof_factorial, 1, CV_32FC1);
                        CvMat *output_mbhx = cvCreateMat(mbh_factorial, 1, CV_32FC1);
                        CvMat *output_mbhy = cvCreateMat(mbh_factorial, 1, CV_32FC1);

                        DETCOM_feature(dest_hog, output_hog, dest_all, 0);
                        DETCOM_feature(dest_hof, output_hof, dest_all, hog_factorial);
                        DETCOM_feature(dest_mbhx, output_mbhx, dest_all, hog_factorial + hof_factorial);
                        DETCOM_feature(dest_mbhy, output_mbhy, dest_all, hog_factorial + hof_factorial + mbh_factorial);
                        //cout<<"!!!!!!!!!!!!!!!!!!!!!!! All Descriptor's Row Size 1 !!!!!!!!!!!!!!!!!!!!!!"<<dest_all.size().height<<endl;
                        //cout<<"!!!!!!!!!!!!!!!!!!!!!!! All Descriptor's Column Size 1 !!!!!!!!!!!!!!!!!!!!!!!!"<<dest_all.size().width<<endl;


                        // ============================== adding part =====================================================
//DETCOM_Eigen_mat
//                      CvMat cv_D_descriptor = dest_all;
                        Mat DETCOM_mat = cvarrToMat(DETCOM_Eigen_mat);
//                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<eigen_t.size().height<<endl;
//                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<eigen_t.size().width<<endl;
//                                                Mat temp_D;
//                                                transpose(cv_D_descriptor, temp_D);
                        Mat DETCOM_Reduction_mat = DETCOM_mat * dest_all;
                        //cout << "!!!!!!!!!!!!!!!!=====t========!!!!!!!!!!!!!!!" << DETCOM_Reduction_mat.size().height << endl;
                        //cout << "!!!!!!!!!!!!!!!!======t=======!!!!!!!!!!!!!!!" << DETCOM_Reduction_mat.size().width << endl;
                        //        Mat temp = zeros(Size(1,8820),CV_8U);
                        //        transpose()


//                                                cout << "!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!" << cv_hog_descriptor.rows << endl;
//                                                cout << "!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!" << cv_hog_descriptor.rows << endl;
                        //        cvTranspose(cv_hog_descriptor, t_hog_descriptor);
                        //        cvMatMul(t_hog_descriptor, Eigen_trans_mat, Reduction_mat);
                        //
                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<hog_descriptor.size().height<<endl;
                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<hog_descriptor.size().width<<endl;

                        //==================================================================================================
//
//                        cout<<"======================== DETCOM_EIGEN_MATRIX ! ======================:"<< DETCOM_mat.at<float>(0,1)<<endl;
//                        cout<<"======================== DETCOM_EIGEN_MATRIX ! ======================:"<< DETCOM_mat.at<float>(1,0)<<endl;
//                        cout<<"======================== DEST_ALL_MATRIX ! ======================:"<< dest_all.at<float>(2,0)<<endl;
//                        cout<<"======================== DEST_ALL_MATRIX ! ======================:"<< dest_all.at<float>(0,2)<<endl;
//                        cout<<"======================== DETCOM_REDUCTION_MATRIX ! ======================:"<< DETCOM_Reduction_mat.at<float>(0,1)<<endl;
//                        cout<<"======================== DETCOM_REDUCTION_MATRIX ! ======================:"<< DETCOM_Reduction_mat.at<float>(1,0)<<endl;

                        first_f_cont[i] = 0;
                        float motion_result = -1, final_result = -1, m_dis[max_classss];

                        clock_t begin, end;
                        begin = clock();

                        motion_result = do_hri_motion_predict(DETCOM_Reduction_mat, motion_dis);
//                        motion_result = do_hri_motion_predict(dest_all, motion_dis);
                        for (int _j = 0; _j < max_classss; _j++) {
                            m_dis[_j] = motion_dis[_j];
                        }
                        //final_result = determin_label(hog_label[i],hog_dis[i],motion_result,m_dis,hog_weight);
                        //action_label_out(motion_result,detcom_map_action3[i]);
//						final_result = determin_label(hog_label[i],hog_dis[i],motion_result,m_dis,hog_weight,moving_jug[i],sum_dis,sum_width,sum_height);
                        final_result = determin_label2(hog_label[i], hog_dis[i], motion_result, m_dis, hog_weight,
                                                       cen_dis[i], id_z_value[i], sum_dis, sum_width, sum_height,
                                                       seq_wid[i], seq_hei[i], detcom_map_action[i]);
                        action_label_out(final_result, detcom_map_action[i]);
//                        				cout<<i<<detcom_map_action[i]<<endl;
                        isoutput[i] = 1;
                        output_label[i] = detcom_map_action[i];
                        cvReleaseMat(&output_hog);
                        cvReleaseMat(&output_hof);
                        cvReleaseMat(&output_mbhx);
                        cvReleaseMat(&output_mbhy);
                        dest_all.release();

                        end = clock();
                        printf("%.6lf\n", ((double)(end-begin)));


//
//
//                            DETCOM_mat.release();
//                            DETCOM_Reduction_mat.release();
//
                    }
                    //                  cout<<"determin part finish"<<endl;
                    moving_jug[i] = false;
                }
                else if (first_f_num[i] >= (frame_ccc / 2) + 1 && second_f_bool[i] == false) {
                    //								cout<<"a2"<<endl;
                    second_f_bool[i] = true;
                    second_f_cont[i] = 0;
                    second_f_num[i] = 0;
                }
                else {
                    first_f_num[i]++;
                    //			cout<<i<<"-th first_f_num[i] : "<<first_f_num[i]++<<endl;
                }
            } // if (first_f_bool[i]) {
            if (second_f_bool[i]) {
                //cout<<"SECOND SLIDING WINDOW STRAT:"<<frame_num<<"\n"<<endl;
                if (second_f_num[i] >= frame_ccc) {
                    //cout<<"second decision : "<< second_f_cont[i] <<endl;
                    second_f_bool[i] = false;
                    first_f_bool[i] = true;
                    second_f_num[i] = 0;

                    //           cout<<"second_f_cont[i] : "<<second_f_cont[i]<<endl;

                    Mat dest_hog, dest_hof, dest_mbhx, dest_mbhy;
                    double sum_dis = 0;
                    moving_jug[i] = moving_jugment(move_dis[i], sum_dis);
                    double sum_width = sum_of_width_distance(move_dis2[i]);
                    //cout << " sum_width : " << sum_width << endl;
                    double sum_height = sum_of_width_distance(move_dis3[i]);
                    //cout << " sum_height : " << sum_height << endl;
                    if (second_f_cont[i] < hofInfo.dim * 3) {
                        no_motion[i] = true;
                        //continue;
                    }
                    else if (second_f_cont[i] < hofInfo.dim * 3) {
                        //            cout<<"under_output"<<endl;
                        no_motion[i] = false;
                        std::vector<float> ext_hog(second_hog_storage[i]);
                        ext_hog.reserve(second_hog_storage[i].size() * 3);
                        ext_hog.insert(ext_hog.end(), second_hog_storage[i].begin(), second_hog_storage[i].end());
                        ext_hog.insert(ext_hog.end(), second_hog_storage[i].begin(), second_hog_storage[i].end());
                        std::vector<float> ext_hof(second_hof_storage[i]);
                        ext_hof.reserve(second_hof_storage[i].size() * 3);
                        ext_hof.insert(ext_hof.end(), second_hof_storage[i].begin(), second_hof_storage[i].end());
                        ext_hof.insert(ext_hof.end(), second_hof_storage[i].begin(), second_hof_storage[i].end());
                        std::vector<float> ext_mbhx(second_mbhx_storage[i]);
                        ext_mbhx.reserve(second_mbhx_storage[i].size() * 3);
                        ext_mbhx.insert(ext_mbhx.end(), second_mbhx_storage[i].begin(), second_mbhx_storage[i].end());
                        ext_mbhx.insert(ext_mbhx.end(), second_mbhx_storage[i].begin(), second_mbhx_storage[i].end());
                        std::vector<float> ext_mbhy(second_mbhy_storage[i]);
                        ext_mbhy.reserve(second_mbhy_storage[i].size() * 3);
                        ext_mbhy.insert(ext_mbhy.end(), second_mbhy_storage[i].begin(), second_mbhy_storage[i].end());
                        ext_mbhy.insert(ext_mbhy.end(), second_mbhy_storage[i].begin(), second_mbhy_storage[i].end());

                        Mat cvt_hog(ext_hog, true);
                        Mat cvt_hof(ext_hof, true);
                        Mat cvt_mbhx(ext_mbhx, true);
                        Mat cvt_mbhy(ext_mbhy, true);

                        Mat data_hog = Mat::zeros(hogInfo.dim * 3, second_f_cont[i], CV_32FC1);
                        Mat data_mbhx = Mat::zeros(mbhInfo.dim * 3, second_f_cont[i], CV_32FC1);
                        Mat data_mbhy = Mat::zeros(mbhInfo.dim * 3, second_f_cont[i], CV_32FC1);
                        Mat data_hof = Mat::zeros(hofInfo.dim * 3, second_f_cont[i], CV_32FC1);

                        int temp_hog = 0, temp_hof = 0;
                        for (int ind_w = 0; ind_w < second_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hogInfo.dim * 3; ind_h++) {
                                data_hog.at<float>(ind_h, ind_w) = cvt_hog.at<float>(temp_hog, 0);
                                data_mbhx.at<float>(ind_h, ind_w) = cvt_mbhx.at<float>(temp_hog, 0);
                                data_mbhy.at<float>(ind_h, ind_w) = cvt_mbhy.at<float>(temp_hog++, 0);
                            }
                        }
                        for (int ind_w = 0; ind_w < second_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hofInfo.dim * 3; ind_h++) {
                                data_hof.at<float>(ind_h, ind_w) = cvt_hof.at<float>(temp_hof++, 0);
                            }
                        }

                        dest_hog = Mat(data_hog).clone().t();
                        second_hog_storage[i].clear();

                        dest_hof = Mat(data_hof).clone().t();
                        second_hof_storage[i].clear();

                        dest_mbhx = Mat(data_mbhx).clone().t();
                        second_mbhx_storage[i].clear();

                        dest_mbhy = Mat(data_mbhy).clone().t();
                        second_mbhy_storage[i].clear();

                        cvt_hog.release();
                        cvt_hof.release();
                        cvt_mbhx.release();
                        cvt_mbhy.release();
                        data_hog.release();
                        data_hof.release();
                        data_mbhx.release();
                        data_mbhy.release();
                    }
                    else {
                        //             cout<<"output"<<endl;
                        no_motion[i] = false;
                        Mat cvt_hog(second_hog_storage[i], true);
                        Mat cvt_hof(second_hof_storage[i], true);
                        Mat cvt_mbhx(second_mbhx_storage[i], true);
                        Mat cvt_mbhy(second_mbhy_storage[i], true);

                        Mat data_hog = Mat::zeros(hogInfo.dim * 3, second_f_cont[i], CV_32FC1);
                        Mat data_mbhx = Mat::zeros(mbhInfo.dim * 3, second_f_cont[i], CV_32FC1);
                        Mat data_mbhy = Mat::zeros(mbhInfo.dim * 3, second_f_cont[i], CV_32FC1);
                        Mat data_hof = Mat::zeros(hofInfo.dim * 3, second_f_cont[i], CV_32FC1);

                        int temp_hog = 0, temp_hof = 0;
                        for (int ind_w = 0; ind_w < second_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hogInfo.dim * 3; ind_h++) {
                                data_hog.at<float>(ind_h, ind_w) = cvt_hog.at<float>(temp_hog, 0);
                                data_mbhx.at<float>(ind_h, ind_w) = cvt_mbhx.at<float>(temp_hog, 0);
                                data_mbhy.at<float>(ind_h, ind_w) = cvt_mbhy.at<float>(temp_hog++, 0);
                            }
                        }
                        for (int ind_w = 0; ind_w < second_f_cont[i]; ind_w++) {
                            for (int ind_h = 0; ind_h < hofInfo.dim * 3; ind_h++) {
                                data_hof.at<float>(ind_h, ind_w) = cvt_hof.at<float>(temp_hof++, 0);
                            }
                        }

                        dest_hog = Mat(data_hog).clone().t();
                        second_hog_storage[i].clear();

                        dest_hof = Mat(data_hof).clone().t();
                        second_hof_storage[i].clear();

                        dest_mbhx = Mat(data_mbhx).clone().t();
                        second_mbhx_storage[i].clear();

                        dest_mbhy = Mat(data_mbhy).clone().t();
                        second_mbhy_storage[i].clear();
                        cvt_hog.release();
                        cvt_hof.release();
                        cvt_mbhx.release();
                        cvt_mbhy.release();
                        data_hog.release();
                        data_hof.release();
                        data_mbhx.release();
                        data_mbhy.release();
                    }
                    //			no_motion[i] = true;
                    if (no_motion[i]) {
                        //				cout<<"["<<i<<"] no_motion output!!!"<<endl;
                        float result = -1;
                        //				result = determin_hog_label2(hog_label[i],hog_dis[i],moving_jug[i]);
                        //				cout<<"resutl : "<<result<<endl;
                        pre_action[i] = detcom_map_action[i];
                        result = determin_hog_label3(hog_label[i], hog_dis[i], seq_wid[i], seq_hei[i], id_z_value[i]);
                        action_label_out(result, detcom_map_action[i]);
                        isoutput[i] = 1;
                        output_label[i] = detcom_map_action[i];
                        //cout << detcom_map_action[i] << endl;
                    }
                    else {
                        //			cout<<"{"<<i<<"} Motion output"<<endl;
                        Mat dest_all = Mat::zeros(hog_factorial + hof_factorial + mbh_factorial + mbh_factorial, 1,
                                                  CV_32FC1);
                        CvMat *output_hog = cvCreateMat(hog_factorial, 1, CV_32FC1);
                        CvMat *output_hof = cvCreateMat(hof_factorial, 1, CV_32FC1);
                        CvMat *output_mbhx = cvCreateMat(mbh_factorial, 1, CV_32FC1);
                        CvMat *output_mbhy = cvCreateMat(mbh_factorial, 1, CV_32FC1);

                        DETCOM_feature(dest_hog, output_hog, dest_all, 0);
                        DETCOM_feature(dest_hof, output_hof, dest_all, hog_factorial);
                        DETCOM_feature(dest_mbhx, output_mbhx, dest_all, hog_factorial + hof_factorial);
                        DETCOM_feature(dest_mbhy, output_mbhy, dest_all, hog_factorial + hof_factorial + mbh_factorial);

                        //cout<<"!!!!!!!!!!!!!!!!!!!!!!! All Descriptor's Row Size 1 !!!!!!!!!!!!!!!!!!!!!!"<<dest_all.size().height<<endl;
                        //cout<<"!!!!!!!!!!!!!!!!!!!!!!! All Descriptor's Column Size 1 !!!!!!!!!!!!!!!!!!!!!!!!"<<dest_all.size().width<<endl;




                        // ============================== adding part =====================================================
//DETCOM_Eigen_mat
//                      CvMat cv_D_descriptor = dest_all;
                        Mat DETCOM_mat = cvarrToMat(DETCOM_Eigen_mat);
//                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<eigen_t.size().height<<endl;
//                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<eigen_t.size().width<<endl;
//                                                Mat temp_D;
//                                                transpose(cv_D_descriptor, temp_D);
                        Mat DETCOM_Reduction_mat = DETCOM_mat * dest_all;
                        //cout << "!!!!!!!!!!!!!!!!=====t_========!!!!!!!!!!!!!!!" << DETCOM_Reduction_mat.size().height << endl;
                        //cout << "!!!!!!!!!!!!!!!!======t_=======!!!!!!!!!!!!!!!" << DETCOM_Reduction_mat.size().width << endl;
                        //        Mat temp = zeros(Size(1,8820),CV_8U);
                        //        transpose()


//                                                cout << "!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!" << cv_hog_descriptor.rows << endl;
//                                                cout << "!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!" << cv_hog_descriptor.rows << endl;
                        //        cvTranspose(cv_hog_descriptor, t_hog_descriptor);
                        //        cvMatMul(t_hog_descriptor, Eigen_trans_mat, Reduction_mat);
                        //
                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<hog_descriptor.size().height<<endl;
                        //        cout<<"!!!!!!!!!!!!!!!!=============!!!!!!!!!!!!!!!"<<hog_descriptor.size().width<<endl;

                        //==================================================================================================
//                          cout<<"======================== DETCOM_EIGEN_MATRIX ======================:"<< DETCOM_mat.at<float>(0,1)<<endl;
//                          cout<<"======================== DETCOM_EIGEN_MATRIX ======================:"<< DETCOM_mat.at<float>(1,0)<<endl;

                        first_f_cont[i] = 0;
                        float motion_result = -1, final_result = -1, m_dis[max_classss];

                        motion_result = do_hri_motion_predict(DETCOM_Reduction_mat, motion_dis);
//                        motion_result = do_hri_motion_predict(dest_all, motion_dis);
                        for (int _j = 0; _j < max_classss; _j++) {
                            m_dis[_j] = motion_dis[_j];
                        }
                        pre_action[i] = detcom_map_action[i];
                        //final_result = determin_label(hog_label[i],hog_dis[i],motion_result,m_dis,hog_weight);
                        //action_label_out(motion_result,detcom_map_action3[i]);
                        //final_result = determin_label(hog_label[i],hog_dis[i],motion_result,m_dis,hog_weight,moving_jug[i],sum_dis,sum_width,sum_height);
                        final_result = determin_label2(hog_label[i], hog_dis[i], motion_result, m_dis, hog_weight,
                                                       cen_dis[i], id_z_value[i], sum_dis, sum_width, sum_height,
                                                       seq_wid[i], seq_hei[i], detcom_map_action[i]);
                        action_label_out(final_result, detcom_map_action[i]);

                        cvReleaseMat(&output_hog);
                        cvReleaseMat(&output_hof);
                        cvReleaseMat(&output_mbhx);
                        cvReleaseMat(&output_mbhy);
                        dest_all.release();
//
//                            DETCOM_mat.release();
//                            DETCOM_Reduction_mat.release();
//
                        isoutput[i] = 1;
                        output_label[i] = detcom_map_action[i];
                        //					cout<<detcom_map_action[i]<<endl;
                    }
                    moving_jug[i] = false;
                }
                else if (second_f_num[i] >= (frame_ccc / 2) + 1 && first_f_bool[i] == false) {
                    first_f_bool[i] = true;
                    first_f_cont[i] = 0;
                    first_f_num[i] = 0;
                }
                else {
                    second_f_num[i]++;
                    //			cout<<i<<"-th second_f_num[i] : "<<second_f_num[i]++<<endl;
                }
            }
            CvScalar color_cvrgb[10] = {CV_RGB(255, 255, 255), CV_RGB(255, 0, 255), CV_RGB(0, 255, 255),
                                        CV_RGB(255, 0, 0), CV_RGB(0, 255, 0)};
            cvRectangle(frame_rgb_copy, pt1[i], pt2[i], color_cvrgb[i], 1, 8, 0);
            //      cvRectangle(frame_depth_copy, pt1[i], pt2[i], color_cvrgb[i], 1, 8, 0);
            CvFont font;
            double hScale = 1;
            double vScale = 1;
            int lineWidth = 3;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);
            char ouput_text[30], ouput_text2[30];
            snprintf(ouput_text2, 30, "[%d]%s", i, detcom_map_action[i].c_str());
//			snprintf(ouput_text2,30,"%s",detcom_map_action[i].c_str());
            //       cout<<i<<" : "<<ouput_text2<<endl;
//            cvCircle(frame_rgb_copy, {100,300},8, CV_RGB(255,0,0));
            print_text(frame_rgb_copy, ouput_text2, pt3[i], font, CV_RGB(0, 255, 0));
            /*  cout<<"move_dis[i][circle_hog[i]-1] : "<<move_dis[i][circle_hog[i]-1]<<endl;
            if(move_dis[i][circle_hog[i]-1]>=200)
            {
            	peo_change_cnt[i] = 10;
            }
            if(peo_change_cnt[i] < 2)
            {
            	peo_change_cnt[i]--;
				snprintf(ouput_text2,30,"[%d]%s",i,detcom_map_action[i].c_str());
		 //       cout<<i<<" : "<<ouput_text2<<endl;
	//            cvCircle(frame_rgb_copy, {100,300},8, CV_RGB(255,0,0));
				print_text(frame_rgb_copy,ouput_text2,pt3[i],font,CV_RGB(0,255,0));
            }
            else
            {
            	peo_change_cnt[i]--;
            }*/
            if (circle_hog[i] >= frame_ccc) {
                circle_hog[i] = circle_hog[i] % frame_ccc;
            }
        }
        CvFont font;
        double hScale = 1;
        double vScale = 1;
        int lineWidth = 3;
        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);
        if (_total >= 1) {
            draw_fight(frame_rgb_copy, frame_depth_copy, _total, detcom_map_action, font, pt1, pt2, id_z_value);
            draw_violence(frame_rgb_copy, frame_depth_copy, _total, detcom_map_action, font, pt1, pt2, id_z_value);
        }
/*        if(_total >=1)
        draw_walk_toge(frame_rgb_copy,frame_depth_copy,_total,pre_action,detcom_map_action,font,pt1, pt2,id_z_value);
        */
        init_counter = 0;
        grey.copyTo(prev_grey);
        for (int i = 0; i < scale_num; i++) {
            grey_pyr[i].copyTo(prev_grey_pyr[i]);
            poly_pyr[i].copyTo(prev_poly_pyr[i]);
        }
        frame_num++;
//        end = clock();
//        printf("%f\n", ((double)(end-begin))/CLOCKS_PER_SEC);
//       printf("%d FRAME COMPUTATION TIME: %f\n",frame_num, ((double)(end-begin))/CLOCKS_PER_SEC);
//        cout<<"FRAME COMPUTATION TIME: "<<((end-begin)/(double(1000)))<<"second(s)."<<endl;

        //       cvShowImage("RGB Window F",frame_rgb_copy);

        //       end2 = clock();
        //       printf("iter Time : %f\n", ((double)(end2-str2)) / CLOCKS_PER_SEC);
        cout<<"frame Num : "<< frame_num<<endl;
    } // else
    // ================================================================================
//	cvShowImage("Depth Window",frame_depth_copy);

    //cvReleaseImage(&frame_rgb_copy);
    //cvReleaseImage(&frame_gray);
//	cv::waitKey(3);

    //   cvReleaseImage(&frame_rgb_copy);
    //cout<<"recognization OUT: "<<frame_num<<endl;

}

void gr_module_4y::stop() {

    bool user_exit = false;
    while (ros::ok() || user_exit) {
        char key;
        key = cvWaitKey(30);
        ros::spinOnce();

        if (tolower(key) == 'q') {
            ROS_INFO("user exit!");
            user_exit = true;
            break;
        }
        if (tolower(key) == 'c') {
//			classfication();
        }
    }
}
