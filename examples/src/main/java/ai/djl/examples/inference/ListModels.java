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
package ai.djl.examples.inference;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ListModels {

    private static final Logger logger = LoggerFactory.getLogger(ListModels.class);

    private ListModels() {
    }

    private static URI BASE_URI = URI.create("https://mlrepo.djl.ai/");

    public static void main(String[] args) throws IOException, ModelNotFoundException {
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        models.forEach(
                (app, list) -> {
                    String appName = app.toString();
                    list.forEach(
                            artifact -> {
                                logger.info("Application:{} Model:{}", appName, artifact);
                                logger.info("This model contains the following files:");
                                for (Map.Entry<String, Artifact.Item> entry :
                                        artifact.getFiles().entrySet()) {
                                    logger.info("file: " + entry.getKey());
                                    URI fileUri = URI.create(entry.getValue().getUri());
                                    URI baseUri = artifact.getMetadata().getRepositoryUri();
                                    if (!fileUri.isAbsolute()) {
                                        fileUri = BASE_URI.resolve(baseUri).resolve(fileUri);
                                    }
                                    logger.info("download link: " + fileUri.toString());
                                }
                            });
                });
    }
}
