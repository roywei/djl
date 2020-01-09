/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

package ai.djl.examples;

import ai.djl.Device;
import ai.djl.examples.training.transferlearning.TrainResnetWithCifar10;
import ai.djl.examples.training.util.ExampleTrainingResult;
import ai.djl.repository.zoo.ModelNotFoundException;
import java.io.IOException;
import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrainResNetTest {

    @Test
    public void testTrainResNet() throws ParseException, ModelNotFoundException, IOException {
        // Limit max 4 gpu for cifar10 training to make it converge faster.
        // and only train 10 batch for unit test.
        String[] args = {"-e", "2", "-g", "4", "-m", "10", "-s", "-p"};

        TrainResnetWithCifar10 test = new TrainResnetWithCifar10();
        Assert.assertTrue(test.runExample(args).isSuccess());
    }

    @Test
    public void testTrainResNetSymbolicNightly()
            throws ParseException, ModelNotFoundException, IOException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (Device.getGpuCount() > 0) {
            // Limit max 4 gpu for cifar10 training to make it converge faster.
            // and only train 10 batch for test.
            int numGPU = Math.max(Device.getGpuCount(), 4);
            String[] args = {"-e", "15", "-g", String.valueOf(numGPU), "-s", "-p"};

            TrainResnetWithCifar10 test = new TrainResnetWithCifar10();
            ExampleTrainingResult result = test.runExample(args);
            Assert.assertTrue(result.isSuccess());
            Assert.assertTrue(result.getTrainingAccuracy() > .7f);
            Assert.assertTrue(result.getTrainingLoss() < .8f);
        }
    }

    @Test
    public void testTrainResNetImperativeNightly()
            throws ParseException, ModelNotFoundException, IOException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (Device.getGpuCount() > 0) {
            // Limit max 4 gpu for cifar10 training to make it converge faster.
            // and only train 10 batch for test.
            int numGPU = Math.max(Device.getGpuCount(), 4);
            String[] args = {"-e", "15", "-g", String.valueOf(numGPU)};

            TrainResnetWithCifar10 test = new TrainResnetWithCifar10();
            ExampleTrainingResult result = test.runExample(args);
            Assert.assertTrue(result.isSuccess());
            Assert.assertTrue(result.getTrainingAccuracy() > .7f);
            Assert.assertTrue(result.getTrainingLoss() < .8f);
        }
    }
}
