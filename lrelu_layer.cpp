#include <fstream>
#include <algorithm>
#include <vector>

#include "caffe/layers/lrelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void LReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				   const vector<Blob<Dtype>*>& top) {
  Iter_ = 0;
  Size_ = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    const int size = Size_;
    vector<int> weight_shape(1);
    weight_shape[0] = size;
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    Dtype* a = this->blobs_[0]->mutable_cpu_data();
    for (int i = 0; i < size; i++) {
      a[i] = (Dtype)1.0;
    }
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }
}

template <typename Dtype>
void LReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int size = Size_;
  const Dtype* a = this->blobs_[0]->cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < size; ++j) {
      top_data[i*size + j] = a[j] * std::max(bottom_data[i*size + j], Dtype(0));
    }
  }

  // Logging Weight
  // std::cout << "********************" << std::endl;
  // std::cout << bottom[0]->num() << '\t' << bottom[0]->count() << std::endl;
  // std::cout << this->blobs_[0]->count() << '\t' << this->blobs_[0]->num() << std::endl;
  // std::cout << bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() << std::endl;
  // std::cout << "********************" << std::endl;
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

template <typename Dtype>
void LReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				    const vector<bool>& propagate_down,
				    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
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
      a_diff[i] /= num;
    }
  }

  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int num = bottom[0]->num();
    const int size = Size_;
    const Dtype* a = this->blobs_[0]->cpu_data();
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
