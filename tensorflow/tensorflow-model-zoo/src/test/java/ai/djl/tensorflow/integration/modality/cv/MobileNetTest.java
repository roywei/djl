/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Transform;
import ai.djl.translate.TranslateException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class MobileNetTest {

    @Test
    public void testMobileNetV2() throws IOException, ModelException, TranslateException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Tensorflow doesn't support Windows yet.");
        }

        Criteria<BufferedImage, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(BufferedImage.class, Classifications.class)
                        .optArtifactId("resnet")
                        .optFilter("flavor", "v1")
                        .optProgress(new ProgressBar())
                        .build();


        try (ZooModel<BufferedImage, Classifications> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, Classifications> predictor =
                    model.newPredictor(new MyTranslator())) {
                Classifications result =
                        predictor.predict(
                                BufferedImageUtils.fromFile(
                                        Paths.get("../../examples/src/test/resources/kitten.jpg")));
                System.out.println(result.best());
            }
        }
    }

    private static final class MyTranslator implements Translator<BufferedImage, Classifications> {

        private List<String> classes;

        @Override
        public Batchifier getBatchifier() {
            return null;
        }

        public MyTranslator() {
            classes = IntStream.range(0, 1000).mapToObj(String::valueOf).collect(Collectors.toList());
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            NDArray array = BufferedImageUtils.toNDArray(ctx.getNDManager(), input, NDImageUtils.Flag.COLOR);
            return new NDList(NDImageUtils.resize(array, 224).div(224f).expandDims(0));
        }

        /** {@inheritDoc} */
        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.singletonOrThrow();
            return new Classifications(classes, probabilities);
        }
    }
}
