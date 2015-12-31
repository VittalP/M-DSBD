#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#define BETA 0.9  // hard code beta
namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= ( target[i] == 0 ? 1-BETA : BETA ) * (input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    caffe_copy(count, sigmoid_output_data, bottom_diff); 
    //for (int i = 0; i < count; ++i) {
      //bottom_diff[i] = target[i]*sigmoid_output_data[i]*0.96940; // pn pnhat (2b-1)
    //}
    caffe_gpu_mul<Dtype>(count, target, sigmoid_output_data, bottom_diff);
    caffe_gpu_scal(count, Dtype(2*BETA-1), bottom_diff);
    caffe_gpu_axpy(count, Dtype(1-BETA), sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-BETA), target, bottom_diff);
    
    //caffe_copy(count, sigmoid_output_data, bottom_diff); // sigmoid_output is p hat; target is p
    //caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    //LOG(INFO) << "top0-mutablecpudata = " << top[0]->mutable_cpu_data()[0] << " loss_weight = " << loss_weight;
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
