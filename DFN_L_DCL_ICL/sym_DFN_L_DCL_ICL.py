import mxnet as mx
import DFN_L_DCL_ICL.config as cfg
import numpy as np

def get_DFN_module(conv_feat):
    deform_bn = mx.symbol.BatchNorm(name='deform_bn', data=conv_feat, use_global_stats=True,
                                        fix_gamma=False, eps=2e-5)
    deform_relu = mx.symbol.Activation(name='deform_relu', data=deform_bn, act_type='relu')
    deform_offset = mx.symbol.Convolution(name='deform_offset_offset', data = deform_relu,
                                                  num_filter=18, pad=(1, 1), kernel=(3, 3), stride=(1, 1), cudnn_off=True)
    deform = mx.contrib.symbol.DeformableConvolution(name='deform_conv', data=conv_feat, offset=deform_offset,
                                                             num_filter=64, pad=(2, 2), kernel=(3, 3), num_deformable_group=1,
                                                             stride=(1, 1), dilate=(2, 2), no_bias=True) 
   
    index_x = mx.symbol.arange(start=0,stop=18,step=2,dtype=np.int32)
    index_y = mx.symbol.arange(start=1,stop=19,step=2,dtype=np.int32)
    deform_offset_t=mx.symbol.transpose(deform_offset,axes=(1,0,2,3))

    offset_x = mx.symbol.take(name='deform_offset_x',a=deform_offset_t,axis=0,indices=index_x)
    offset_y = mx.symbol.take(name='deform_offset_y',a=deform_offset_t,axis=0,indices=index_y)
    offset_x = mx.symbol.transpose(offset_x,axes=(1,0,2,3))  
    offset_y = mx.symbol.transpose(offset_y,axes=(1,0,2,3))

    offset_xmean = mx.symbol.mean(name='deform_offset_xmean',data=offset_x,axis=1,keepdims=1)
    offset_ymean = mx.symbol.mean(name='deform_offset_ymean',data=offset_y,axis=1,keepdims=1)
    offset_xmean = mx.symbol.repeat(data=offset_xmean,axis=1,repeats=9)
    offset_ymean = mx.symbol.repeat(data=offset_ymean,axis=1,repeats=9)

    x_residual=offset_x - offset_xmean
    y_residual=offset_y - offset_ymean
    x_residual=mx.symbol.Flatten(data=x_residual)
    y_residual=mx.symbol.Flatten(data=y_residual)
    x_residual=x_residual*x_residual
    y_residual=y_residual*y_residual

    x_residual=mx.symbol.sum(x_residual,axis=1,keepdims=1)
    x_residual=x_residual/8649#31*31*9
    y_residual=mx.symbol.sum(y_residual,axis=1,keepdims=1)
    y_residual=y_residual/8649#31*31*9
    
    dcl_loss = x_residual + y_residual
  
    fc1_normed = mx.symbol.L2Normalization(deform, mode='instance', name='L2norm')
    anchor = mx.symbol.slice_axis(fc1_normed, axis=0, begin=0, end=cfg.batch_size//2)
    anchor = mx.symbol.Flatten(data=anchor)
    positive = mx.symbol.slice_axis(fc1_normed, axis=0, begin=cfg.batch_size//2, end=cfg.batch_size)
    positive = mx.symbol.Flatten(data=positive)
    ap = anchor - positive
    ap = ap*ap   
    icl_loss = mx.symbol.sum(ap, axis=1, keepdims=1)               
    return dcl_loss,icl_loss,deform

def net_unit(data, num_filter, stride, dim_match, name, bn_mom=0.99):
    bn2 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter), kernel=(3,3), stride=stride, pad=(1,1),
                               no_bias=True, name=name + '_conv2')    
    return conv2

def DFN_L(units, num_stages, filter_list, num_classes, bn_mom=0.99):
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')

    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(4,4), pad=(3, 3),
                              no_bias=True, name="conv0")
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    dcl_loss,icl_loss, body = get_DFN_module(body)
    for i in range(num_stages):
        stride = 2
        body = net_unit(body, filter_list[i+1], (stride, stride), False,
                             name='stage%d_unit%d' % (i + 1, 1))
        for j in range(units[i]-1):
            body = net_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2))
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')

    pool1 = mx.symbol.Pooling(data=relu1, kernel=(4, 4), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)

    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=1024, name='fc1')
    fc1_bn = mx.sym.BatchNorm(data=fc1, fix_gamma=False, eps=2e-5, momentum=0.99, name='fc1_bn')
    fc1_act = mx.sym.Activation(data=fc1_bn, act_type='relu', name='fc1_act')
    fc2 = mx.symbol.FullyConnected(data=fc1_act, num_hidden=num_classes, name='fc2')
    
    symbol=mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    extra_loss_1 = mx.symbol.MakeLoss(icl_loss,grad_scale=0.01)
    extra_loss_2 = mx.symbol.MakeLoss(dcl_loss,grad_scale=0.001)
    stu_out = mx.symbol.Group([symbol, extra_loss_1, extra_loss_2])  
    return stu_out 

def get_symbol(num_classes, num_layers=8):
    filter_list = [64, 64, 128]

    num_stages = 2
    if num_layers == 8:
        units = [2,2]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return DFN_L(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes)
