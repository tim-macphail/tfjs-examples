/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import { BostonHousingDataset, featureDescriptions } from "./data";
import * as normalization from "./normalization";
import * as ui from "./ui";

// Some hyperparameters for model training.
const NUM_EPOCHS = 20;
const BATCH_SIZE = 40;
const LEARNING_RATE = 0.01;

const bostonData = new BostonHousingDataset();
const tensors = {};

// Convert loaded data into tensors and creates normalized versions of the
// features.
export function arraysToTensors() {
  tensors.rawTrainFeatures = tf.tensor2d(bostonData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(bostonData.trainTarget);
  tensors.rawTestFeatures = tf.tensor2d(bostonData.testFeatures);
  tensors.testTarget = tf.tensor2d(bostonData.testTarget);
  // Normalize mean and standard deviation of data.
  let { dataMean, dataStd } = normalization.determineMeanAndStddev(
    tensors.rawTrainFeatures
  );

  tensors.trainFeatures = normalization.normalizeTensor(
    tensors.rawTrainFeatures,
    dataMean,
    dataStd
  );
  tensors.testFeatures = normalization.normalizeTensor(
    tensors.rawTestFeatures,
    dataMean,
    dataStd
  );
}

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
export function linearRegressionModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [bostonData.numFeatures], units: 1 })
  );

  return model;
}

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 1 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export function multiLayerPerceptronRegressionModel1Hidden() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [bostonData.numFeatures],
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );
  model.add(tf.layers.dense({ units: 1 }));

  return model;
}

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
export function multiLayerPerceptronRegressionModel2Hidden() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [bostonData.numFeatures],
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );
  model.add(
    tf.layers.dense({
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal",
    })
  );
  model.add(tf.layers.dense({ units: 1 }));

  // model.summary();
  return model;
}

/**
 * Describe the current linear weights for a human to read.
 *
 * @param {Array} kernel Array of floats of length 12.  One value per feature.
 * @returns {List} List of objects, each with a string feature name, and value
 *     feature weight.
 */
export function describeKernelElements(kernel) {
  tf.util.assert(
    kernel.length == 12,
    `kernel must be a array of length 12, got ${kernel.length}`
  );
  const outList = [];
  for (let idx = 0; idx < kernel.length; idx++) {
    outList.push({ description: featureDescriptions[idx], value: kernel[idx] });
  }
  return outList;
}

/**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epcoh.
 *
 * @param {tf.Sequential} model Model to be trained.
 * @param {boolean} weightsIllustration Whether to print info about the learned
 *  weights.
 */
export async function run(model, modelName, weightsIllustration) {
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });

  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);

  ui.updateStatus("Starting training process...");
  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    // callbacks: {
    //   onEpochEnd: async (epoch, logs) => {
    //     await ui.updateModelStatus(
    //       `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`,
    //       modelName
    //     );
    //     trainLogs.push(logs);
    //     tfvis.show.history(container, trainLogs, ["loss", "val_loss"]);

    //     if (weightsIllustration) {
    //       model.layers[0]
    //         .getWeights()[0]
    //         .data()
    //         .then((kernelAsArr) => {
    //           const weightsList = describeKernelElements(kernelAsArr);
    //           ui.updateWeightDescription(weightsList);
    //         });
    //     }
    //   },
    // },
  });

  ui.updateStatus("Done!");

  console.log(document.getElementById("flow").value);

  const flow = parseFloat(document.getElementById("flow").value);
  const waterlevel = parseFloat(document.getElementById("waterlevel").value);

  const output = model.predict(tf.tensor2d([[waterlevel, flow]]));
  const prediction = Array.from(output.dataSync())[0];
  console.log({ prediction });
}

export function computeBaseline() {
  const avgPrice = tensors.trainTarget.mean();
  const baseline = tensors.testTarget.sub(avgPrice).square().mean();
  const baselineMsg = `Baseline loss (meanSquaredError) is ${baseline
    .dataSync()[0]
    .toFixed(2)}`;
  ui.updateBaselineStatus(baselineMsg);
}

document.addEventListener(
  "DOMContentLoaded",
  async () => {
    await bostonData.loadData();
    ui.updateStatus("Data loaded, converting to tensors");
    arraysToTensors();
    ui.updateStatus(
      "Data is now available as tensors.\n" + "Click a train button to begin."
    );
    // TODO Explain what baseline loss is. How it is being computed in this
    // Instance
    ui.updateBaselineStatus("Estimating baseline loss");
    computeBaseline();
    await ui.setup();
  },
  false
);
