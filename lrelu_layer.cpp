#include <fstream>
#include <algorithm>
#include <vector>

#include "caffe/layers/lrelu_layer.hpp"

namespace caffe {


// init
template <typename Dtype>
void LReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				   const vector<Blob<Dtype>*>& top) {
  Iter_ = 0;
  Size_ = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Initializing the weights
    const int size = Size_;
    vector<int> weight_shape(1);
    weight_shape[0] = size; // size of the parameters of lrelu, one dimension
    this->blobs_.resize(1); // without bias
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape)); // alloc the memory
    Dtype* a = this->blobs_[0]->mutable_cpu_data(); // initialization the value with 1.0
    for (int i = 0; i < size; i++) {
      a[i] = (Dtype)1.0;
    }
    this->param_propagate_down_.resize(this->blobs_.size(), true); // needs backward to optimize
  }
}

// use bottom and weigths to get top.
template <typename Dtype>
void LReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data(); // data that readable
  Dtype* top_data = top[0]->mutable_cpu_data(); // data that writable
  const int num = bottom[0]->num(); // batch size
  const int size = Size_; // size that when batch size = 1
  const Dtype* a = this->blobs_[0]->cpu_data(); // weights that readble
  for (int i = 0; i < num; ++i) { // every image in the data set, batch size in total
    for (int j = 0; j < size; ++j) { // every value in one image
      top_data[i*size + j] = a[j] * std::max(bottom_data[i*size + j], Dtype(0)); // i means i-th image, j means j-th value in i-th image
    }
  }

  // Logging Weight
  ++Iter_;
  if ((Iter_+1) % 2999 == 0) {
    const char* s1 = "/home/ddstone/Development/workspace/caffe/log/cifar10_weights/";
    const char* s2 = ".txt";
    char name[100];
    sprintf(name, "%s%d%s", s1, Iter_+1, s2);
    std::ofstream f1(name);
    for (int i = 0; i < size; ++i) {
      f1 << a[i] << " ";
    }
    f1.close();
  }
}

// use top (top_diff) and weights to get bottom_diff
template <typename Dtype>
void LReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				    const vector<bool>& propagate_down,
				    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) { // if the weights need backward
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* a_diff = this->blobs_[0]->mutable_cpu_diff();
    const int num = bottom[0]->num();
    const int size = Size_;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < size; ++j) {
        a_diff[j] += top_diff[i*size + j] * std::max(bottom_data[i*size + j], Dtype(0));
      }
    }
    for (int i = 0; i < size; ++i) {
      a_diff[i] /= num; // i means i-th a_diff, to average the gradient, for adding 'num'(batch_size) times above
    }
  }

  if (propagate_down[0]) { // if the data need backward
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int num = bottom[0]->num();
    const int size = Size_;
    const Dtype* a = this->blobs_[0]->cpu_data();
    // consider it as the reverse progress of what happens in forward function
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < size; ++j) {
        bottom_diff[i*size + j] = top_diff[i*size + j] * (bottom_data[i*size + j] > 0) * a[j];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LReLULayer);
#endif

INSTANTIATE_CLASS(LReLULayer);
REGISTER_LAYER_CLASS(LReLU);

}  // namespace caffe
