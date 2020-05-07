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

package ai.djl.tensorflow.integration.modality.cv;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import ai.djl.engine.Engine;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDManager;
import ai.djl.tensorflow.engine.TfNDArray;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.MetaGraphDef;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;
import org.testng.annotations.Test;

public class TensorFlowTest {

    @Test
    @SuppressWarnings({"unchecked", "rawtypes"})
    public <T extends TType> void testTensorFlow() throws IOException {
        String imageFile = "../../examples/src/test/resources/kitten.jpg";
        String modelFile = "/Users/lawei/.djl.ai/cache/repo/model/cv/image_classification/ai/djl/tensorflow/mobilenet/v2/imagenet/0.0.1/";
        String labelFile = "/Users/lawei/Downloads/test/ImageNetLabels.txt";
        // read image
        NDManager manager = Engine.getInstance().newBaseManager();
        TfNDArray image = (TfNDArray) BufferedImageUtils.toNDArray(manager, BufferedImageUtils.fromFile(Paths.get(imageFile)));

        image = (TfNDArray) NDImageUtils.resize(image, 224).div(127.5f).sub(1f).expandDims(0);
        // byte[] imageBytes = readAllBytesOrExit(Paths.get(args[0]));
        EagerSession eagerSession = EagerSession.options().async(true).build();
        Ops tf = Ops.create(eagerSession);
        Operation dataop =
                eagerSession
                        .opBuilder("Const", "Const")
                        .setAttr("dtype", image.getTensor().dataType())
                        .setAttr("value", image.getTensor())
                        .build();
        Operand<T> data = dataop.output(0);

        // load model
        SavedModelBundle.Loader loader = SavedModelBundle.loader(modelFile).withTags("serve");
        SavedModelBundle bundle = loader.load();
        Session session = bundle.session();

        // parse graph definition
        MetaGraphDef metaGraphDef = bundle.metaGraphDef();

        // extract input and output name from graph definition
        Map<String, SignatureDef> signatureDefMap = metaGraphDef.getSignatureDefMap();
        SignatureDef servingDefault =
                metaGraphDef.getSignatureDefOrDefault(
                        "serving_default",
                        signatureDefMap.get(signatureDefMap.keySet().toArray()[1]));
        String inputKey = (String) servingDefault.getInputsMap().keySet().toArray()[0];
        String inputName = servingDefault.getInputsMap().get(inputKey).getName();
        String outputKey = (String) servingDefault.getOutputsMap().keySet().toArray()[0];
        String outputName = servingDefault.getOutputsMap().get(outputKey).getName();

        Session.Runner runner = session.runner();

        runner.feed(inputName, data.asOutput().tensor());
        runner.fetch(outputName);
        Tensor result = runner.run().get(0);

        // convert tensor back to operand
        Operation op =
                eagerSession
                        .opBuilder("Const", "Const")
                        .setAttr("dtype", result.dataType())
                        .setAttr("value", result)
                        .build();
        Operand operand = op.output(0);
        Tensor argMax = tf.math.argMax(operand, tf.constant(1)).asOutput().tensor();
        long[] index = new long[1];
        argMax.rawData().asLongs().read(index);

        System.out.println("Prediction is " + readLabels(labelFile)[(int) index[0]]);

        eagerSession.close();
        argMax.close();
        result.close();
        session.close();
        bundle.close();
    }

    private static String[] readLabels(String path) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(path), "utf-8");
        List<String> lines = new ArrayList<>();
        while (sc.hasNextLine()) {
            lines.add(sc.nextLine());
        }
        return lines.toArray(new String[0]);
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }
}
