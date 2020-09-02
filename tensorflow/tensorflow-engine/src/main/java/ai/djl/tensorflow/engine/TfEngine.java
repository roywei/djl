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
package ai.djl.tensorflow.engine;

import static org.tensorflow.internal.c_api.global.tensorflow.*;
import static org.tensorflow.internal.c_api.global.tensorflow.TFE_NewContextOptions;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.EagerSession;
import org.tensorflow.TensorFlow;
import org.tensorflow.internal.c_api.TF_DeviceList;
import org.tensorflow.internal.c_api.TF_Status;
import org.tensorflow.internal.c_api.global.*;

/**
 * The {@code TfEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://www.tensorflow.org/">Tensorflow Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code TfEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "TensorFlow".
 */
public final class TfEngine extends Engine {

    private static final Logger logger = LoggerFactory.getLogger(TfEngine.class);

    public static final String ENGINE_NAME = "TensorFlow";

    private TfEngine() {}

    static TfEngine newInstance() {
        try {
            LibUtils.loadLibrary();
            EagerSession.getDefault();

            return new TfEngine();
        } catch (Throwable e) {
            logger.warn("Failed load TensorFlow native library.", e);
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new TfModel(name, device);
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return TensorFlow.version();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        if (StandardCapabilities.MKL.equals(capability)) {
            return true;
        } else if (StandardCapabilities.CUDA.equals(capability)) {
            TF_Status status = TF_NewStatus();
            TF_DeviceList deviceList =
                    TFE_ContextListDevices(TFE_NewContext(TFE_NewContextOptions(), status), status);
            int deviceCount = TF_DeviceListCount(deviceList);
            for (int i = 0; i < deviceCount; i++) {
                if (TF_DeviceListName(deviceList, i, status)
                        .getString()
                        .toLowerCase()
                        .contains("gpu")) {
                    return true;
                }
            }
            return false;
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return TfNDManager.getSystemManager().newSubManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return TfNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("TensorFlow does not support training yet");
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        TfNDManager.setRandomSeed(seed);
    }
}
