import * as tf from '@tensorflow/tfjs';
//const {tf} = require('@tensorflow/tfjs-node')
import * as tfvis from '@tensorflow/tfjs-vis';

// We use a sequential model for linear regression
const model = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [200]}));
model.add(tf.layers.dense({units: 1}));

// Select loss and optimizer for model
model.compile({loss: 'meanSquaredError', optimizer: 'sgd',metrics: ['MSE']});


const xs = tf.randomUniform([10000, 200]);
const ys = tf.randomUniform([10000, 1]);
const valXs = tf.randomUniform([1000, 200]);
const valYs = tf.randomUniform([1000, 1]);

const surface = { name: 'show.fitCallbacks', tab: 'Training' };
// Training the model
model.fit(xs, ys, {epochs: 100,
                   validationData: [valXs, valYs],
                           callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc']),
                          }).then(() => {
    // Use model to predict weight for height 6ft
    model.predict(tf.tensor2d([6], [1,1])).print();
});
const surface2 = { name: 'Model Summary', tab: 'Model' };
tfvis.show.modelSummary(surface2, model);
