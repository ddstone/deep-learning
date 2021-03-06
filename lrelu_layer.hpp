#ifndef CAFFE_LRELU_LAYER_HPP_
#define CAFFE_LRELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Learnable Rectified Linear Unit non-linearity @f$ y = \max(0, a * x) @f$.
 */
  template <typename Dtype>
  class LReLULayer : public NeuronLayer<Dtype> {
  public:
    explicit LReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			    const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "LReLU"; }
    
  protected:
    /**
     * @param bottom input Blob vector (length 1)
     *   -# @f$ (N \times C \times H \times W) @f$
     *      the inputs @f$ x @f$
     * @param top output Blob vector (length 1)
     *   -# @f$ (N \times C \times H \times W) @f$
     *      the computed outputs @f$
     *        y = \max(0, a * x)
     */
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			   const vector<Blob<Dtype>*>& top);
    
    /**
     * @brief Computes the error gradient w.r.t. the LReLU inputs.
     *
     * @param top output Blob vector (length 1), providing the error gradient with
     *      respect to the outputs
     *   -# @f$ (N \times C \times H \times W) @f$
     *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
     *      with respect to computed outputs @f$ y @f$
     * @param propagate_down see Layer::Backward. A flag to determine whether 
     *        the weights need to optimize.
     * @param bottom input Blob vector (length 1)
     *   -# @f$ (N \times C \times H \times W) @f$
     *      the inputs @f$ x @f$; Backward fills their diff with
     *      gradients @f$
     *        \frac{\partial E}{\partial x} = \left\{
     *        \begin{array}{lr}
     *            0 & \mathrm{if} \; x \le 0 \				\
     *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
     *        \end{array} \right.
     *      @f$ if propagate_down[0], by default.
     *      If a non-zero negative_slope @f$ \nu @f$ is provided,
     *      the computed gradients are @f$
     *        \frac{\partial E}{\partial x} = \left\{
     *        \begin{array}{lr}
     *            \nu \frac{\partial E}{\partial y} & \mathrm{if} \; x \le 0 \ \
     *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
     *        \end{array} \right.
     *      @f$.
     */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int Iter_; // for weight logging
    int Size_; // size of bottom data when batch size = 1 (regardless of bottom->num())
  };

}  // namespace caffe

#endif  // CAFFE_RELU_LAYER_HPP_
