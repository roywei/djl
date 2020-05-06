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

package ai.djl.modality.cv.translator;

import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;

/** A generic {@link ai.djl.translate.Translator} for Image Classification tasks. */
public class ImageClassificationTranslator extends BaseImageTranslator<Classifications> {

    private String synsetArtifactName;
    private boolean applySoftmax;

    private List<String> synset;

    /**
     * Constructs an Image Classification using {@link Builder}.
     *
     * @param builder the data to build with
     */
    public ImageClassificationTranslator(Builder builder) {
        super(builder);
        this.synsetArtifactName = builder.synsetArtifactName;
        this.applySoftmax = builder.applySoftmax;
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        synset = model.getArtifact(synsetArtifactName, Utils::readLines);
    }

    /** {@inheritDoc} */
    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) throws IOException {
        NDArray probabilitiesNd = list.singletonOrThrow();
        if (applySoftmax) {
            probabilitiesNd = probabilitiesNd.softmax(0);
        }
        return new Classifications(synset, probabilitiesNd);
    }

    /**
     * Creates a builder to build a {@code ImageClassificationTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A Builder to construct a {@code ImageClassificationTranslator}. */
    public static class Builder extends BaseBuilder<Builder> {

        private String synsetArtifactName;
        private boolean applySoftmax;

        Builder() {}

        /**
         * Sets the name of the synset file listing the potential classes for an image.
         *
         * @param synsetArtifactName a file listing the potential classes for an image
         * @return the builder
         */
        public Builder setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
            return this;
        }

        /**
         * Sets whether to apply softmax when processing output. Some models already include softmax
         * in the last layer, so don't apply softmax when processing model output.
         *
         * @param applySoftmax boolean whether to apply softmax
         * @return the builder
         */
        public Builder optApplySoftmax(boolean applySoftmax) {
            this.applySoftmax = applySoftmax;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the {@link ImageClassificationTranslator} with the provided data.
         *
         * @return an {@link ImageClassificationTranslator}
         */
        public ImageClassificationTranslator build() {
            if (synsetArtifactName == null) {
                throw new IllegalArgumentException("You must specify a synset artifact name");
            }
            return new ImageClassificationTranslator(this);
        }
    }
}
