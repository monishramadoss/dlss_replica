import tensorflow as tf



class dlss_RCNN_Cell(tf.keras.Model):
    def __init__(self,input_channel, output_channel, type_op, SCALING):
        super(dlss_RCNN_Cell, self).__init__()
        self.output_channel = output_channel
        self.type_op = type_op
        self.hidden = None
        if(type_op == 'downsample' or type_op == 'bottleneck'):
            self.l1 = tf.keras.Sequential([
                                            tf.keras.layers.Convolution2D(input_channel, (3,3), (1,1)),
                                            tf.keras.layers.LeakyReLU(alpha=0.1)
                                          ])
           
            self.l2 = tf.keras.Sequential([ 
                                            tf.keras.layers.Convolution2D(output_channel * 2, (3,3), (1,1)),
                                            tf.keras.layers.LeakyReLU(alpha=0.1),
                                            tf.keras.layers.Convolution2D(output_channel, (3,3), (1,1)),
                                            tf.keras.layers.LeakyReLU(alpha=0.1)
                                          ])
        if(type_op == 'upsample'):
            self.l1 = tf.keras.Sequential([
                                            tf.keras.layers.UpSampling2D(size=(SCALING, SCALING), interpolation='nearest'),
                                            tf.keras.layers.Convolution2D(input_channel*2, (3,3), (1,1)),
                                            tf.keras.layers.LeakyReLU(alpha=0.1),
                                            tf.keras.layers.Convolution2D(output_channel, (3,3), (1,1)),
                                            tf.keras.layers.LeakyReLU(alpha=0.1)
                                          ])
            
    def call(self, input):
        if(self.type_op == 'downsample' or self.type_op == 'bottleneck'):
            op1 = self.l1(input)
            op2 = self.l2(tf.concat(input, self.hidden), axis=1)
            self.hidden = op2
            return op2
        if(self.type_op == 'upsample'):
            return self.l1(input)
               
    def reset_hidden(self, input, dfac):
        size = list(input.shape)
        size[0] = size[0].value//dfac
        size[1] = size[1].value//dfac
        size[2] = self.output_channel

        self.hidden_size = size
        self.hidden = tf.zeros((size[0], size[1], size[2]))
        print(self.hidden.shape)


class dlss_autoencoder(tf.keras.Model):
    def __init__(self, SCALING):
        super(dlss_autoencoder, self).__init__()
        
        self.pool = tf.keras.layers.MaxPool2D()

        self.d_cell1 = dlss_RCNN_Cell(3, 32, 'downsample', SCALING)
        self.d_cell2 = dlss_RCNN_Cell(32, 43, 'downsample', SCALING)
        self.d_cell3 = dlss_RCNN_Cell(43, 57, 'downsample', SCALING)
        self.d_cell4 = dlss_RCNN_Cell(57, 76, 'downsample', SCALING)
        self.d_cell5 = dlss_RCNN_Cell(76, 101, 'downsample', SCALING)

        self.bottle_cell6 = dlss_RCNN_Cell(101,101, 'bottleneck', SCALING)

        self.u_cell7 = dlss_RCNN_Cell(101, 76, 'upsample', SCALING)
        self.u_cell8 = dlss_RCNN_Cell(32, 57, 'upsample', SCALING)
        self.u_cell9 = dlss_RCNN_Cell(57, 43, 'upsample', SCALING)
        self.u_cell10 = dlss_RCNN_Cell(43, 32, 'upsample', SCALING)
        self.u_cell11 = dlss_RCNN_Cell(32, 3, 'upsample', SCALING)
   

    def call(self, input):
        d1 = self.pool(self.d_cell1(input))
        d2 = self.pool(self.d_cell2(d1))
        d3 = self.pool(self.d_cell3(d2))
        d4 = self.pool(self.d_cell4(d3))
        d5 = self.pool(self.d_cell5(d4))

        b = self.bottle_cell6(d5)

        u5 = self.u_cell7(tf.concat((b, d5), axis=1))
        u4 = self.u_cell8(tf.concat((u5, d5), axis=1))
        u3 = self.u_cell9(tf.concat((u4, d5), axis=1))
        u2 = self.u_cell10(tf.concat((u3, d5), axis=1))
        output = self.u_cell11(tf.concat((u2, d5), axis=1))

        return output

    def reset_hidden(self, input):
        self.inp = input
        self.d_cell1.reset_hidden(self.inp, 1)
        self.d_cell2.reset_hidden(self.inp, 2)
        self.d_cell3.reset_hidden(self.inp, 4)
        self.d_cell4.reset_hidden(self.inp, 8)
        self.d_cell5.reset_hidden(self.inp, 16)

        self.bottle_cell6.reset_hidden(self.inp, dfac=32)

        self.u_cell7.reset_hidden(self.inp, dfac=16)
        self.u_cell8.reset_hidden(self.inp, dfac=8)
        self.u_cell9.reset_hidden(self.inp, dfac=4)
        self.u_cell10.reset_hidden(self.inp, dfac=2)
        self.u_cell11.reset_hidden(self.inp, dfac=1)