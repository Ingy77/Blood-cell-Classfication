Introduction

In recent years, Artificial Intelligence has made significant progress in image analysis, especially in the medical field, enabling faster and more accurate disease diagnosis. Blood cell classification plays an important role in detecting hematological disorders, infections, and immune system abnormalities. These improvements can enhance patient outcomes, particularly in resource-limited healthcare settings. Hematological disorders affect a significant percentage of the global population, with anemia alone impacting approximately 30% of people worldwide. Therefore, blood cell classification can contribute to improving diagnosis, reducing health risks, and supporting safer blood transfusion processes.

Deep learning CNNs have achieved remarkable success in image tasks such as classification and generation, including medical image analysis. Vision Transformers have recently shown state-of-the-art performance in image classification tasks, achieving very high accuracy in blood cell classification [1]. The BloodMNIST dataset, which is part of the MedMNIST collection, is used in this study. It provides labeled images of blood cells across multiple categories and is widely used for evaluating deep learning models.

In this field, several deep learning models are applied to medical image classification. Transfer learning models such as AlexNet, ResNet, and VGG can be used for blood cell classification. These models learn features such as edges, shapes, colors, and textures. Due to the benefits of artificial intelligence in image analysis, various machine learning techniques have been widely adopted [2].

Despite recent progress, several challenges remain. One major issue is that many studies focus only on accuracy without addressing generalization, especially when working with small datasets such as BloodMNIST. Another challenge is class imbalance, where some classes are underrepresented, leading to biased predictions. In addition, there is still an open question regarding the performance comparison between traditional CNN models and newer architectures such as Vision Transformers on small-scale medical datasets.

To address these challenges, this study aims to compare traditional CNN models with modern deep learning and transformer-based models using the BloodMNIST dataset. The goal is to improve classification performance, handle the mentioned challenges, and identify the best-performing model for this task.

ConvNeXt Experience

The ConvNeXt-Tiny model achieved the highest performance among all tested models, with an accuracy of 97.54%. This makes it the best-performing model compared to other CNN-based architectures. The results indicate that it provides highly accurate classification across all blood cell categories. ConvNeXt combines principles from traditional CNNs and transformer-inspired design, which contributes to its strong performance. It is widely used in medical image research and is well-suited for the BloodMNIST dataset.

From the confusion matrix, the model shows strong classification performance for neutrophils (655 correct), platelets (470 correct), and eosinophils (310 correct). These classes have high true positive rates with minimal misclassification. Some errors occur between monocytes and lymphocytes, as well as between IG and neutrophils, due to biological and visual similarity between these cell types. Despite this, the model maintains strong and balanced performance across all classes.

Eosinophils and platelets are perfectly classified, with precision and recall equal to 1.00. Basophils achieve precision of 0.97 and recall of 0.99. Neutrophils achieve precision of 0.97 and recall of 0.99. Lymphocytes achieve precision of 0.97 and recall of 0.96. Monocytes achieve precision of 0.98 and recall of 0.93. IG achieves precision of 0.93 and recall of 0.95. Erythroblasts achieve precision of 0.99 and recall of 0.96. These results show that the model performs strongly across all classes with only minor misclassifications.

The learning curves indicate stable training behavior with no significant overfitting, as there is no large gap between training and validation performance.

Contribution of ConvNeXt

This study presents the implementation and fine-tuning of the ConvNeXt-Tiny model for blood cell classification using the BloodMNIST dataset. The model was adapted to classify eight target classes by modifying the final classification layer and allowing full fine-tuning to learn domain-specific features.

A key contribution is the use of a class-weighted cross-entropy loss function to address dataset imbalance and reduce prediction bias. In addition, an optimized training strategy was applied, combining the Adam optimizer with a learning rate scheduler based on cosine annealing. Early stopping based on validation accuracy was also used to prevent overfitting.

The evaluation framework includes accuracy, balanced accuracy, precision, recall, F1-score, and confusion matrix analysis, providing detailed insight into both overall and class-wise performance.

EfficientNet-B0 Experience

EfficientNet-B0 achieved strong performance with an accuracy of 96.46%. It performed well across most classes, particularly eosinophils and platelets. However, performance was slightly lower in the IG class compared to ConvNeXt. Overall, the model is efficient and reliable but slightly less accurate than ConvNeXt.

Confusion Matrix Analysis

The confusion matrix shows a strong diagonal distribution, indicating correct classification of most samples. The model performs well in eosinophils (621 correct), neutrophils (647 correct), and platelets (470 correct). These classes show high true positive rates with minimal misclassification. However, some errors occur in the IG class, where misclassification happens mainly with monocytes and neutrophils.

Classification Report Analysis

The classification report shows strong performance for EfficientNet-B0. Eosinophils achieve precision of 0.99 and recall of 1.00. Platelets achieve precision of 1.00 and recall of 1.00. Erythroblasts achieve an F1-score of 0.98, and neutrophils achieve an F1-score of 0.97. The lowest performance is observed in the IG class, with precision of 0.92.

Learning Curves Analysis

The learning curves show stable training behavior and no significant overfitting. The loss decreases steadily, indicating that the model learns effectively over time.

Contribution of EfficientNet-B0

This study implements and fine-tunes the EfficientNet-B0 model for blood cell classification using the BloodMNIST dataset. The model is initialized with pretrained weights and modified to match the eight target classes, followed by full fine-tuning.

Improvements include the use of a class-weighted cross-entropy loss function to handle class imbalance, as well as the combination of the AdamW optimizer with a cosine annealing learning rate scheduler. Early stopping was applied to improve convergence and prevent overfitting.

The evaluation framework includes accuracy, balanced accuracy, precision, recall, F1-score, and confusion matrix analysis.

ResNet-18 Experience

ResNet-18 was evaluated to compare its performance with ConvNeXt and EfficientNet. Although it is an older architecture, it remains important due to its simplicity and efficiency.

The confusion matrix shows strong classification performance across most classes. The model performs well in eosinophils (616 correct), neutrophils (642 correct), and platelets (469 correct). However, misclassifications occur between biologically similar classes such as monocytes and neutrophils, and between erythroblasts and other related classes.

Compared to EfficientNet and ConvNeXt, ResNet-18 shows slightly higher misclassification in complex classes.

The learning curves show stable training with no major overfitting. Training accuracy reaches 99%, while validation accuracy reaches 96%, indicating a small degree of overfitting compared to EfficientNet.

Classification Report

The classification report shows strong performance across most classes, with near-perfect precision and recall. However, some classes show minor reductions compared to other models. Despite this, the overall performance remains strong.

Results

ResNet-18 achieved an accuracy of approximately 96% on the BloodMNIST dataset. It is fast and simple but performs slightly worse than ConvNeXt and EfficientNet.

Contribution of ResNet-18
Evaluation of model performance on BloodMNIST
Use of AdamW optimizer with cosine annealing learning rate
Application of early stopping
Use of precision, recall, and F1-score for evaluation
Handling class imbalance during training
Use of pretrained weights with modified final classification layer for eight classes
Vision Transformer (ViT-B/16)

This section presents the evaluation of the Vision Transformer (ViT-B/16) model on the BloodMNIST dataset to compare its performance with CNN-based models.

The confusion matrix shows that most classes are classified with high accuracy. Some minor confusion occurs between visually similar classes such as basophils and eosinophils. Overall, the model demonstrates strong feature extraction capabilities with minimal inter-class confusion.

Classification Report

The classification report evaluates precision, recall, and F1-score. Most classes achieve values above 0.95, indicating strong and reliable performance. High F1-scores demonstrate a good balance between precision and recall. Some variation in recall indicates that certain classes are more difficult to classify.

Overall, the ViT model achieves strong and balanced performance across all classes, making it suitable for medical image classification.

Contribution of ViT Model
Global feature extraction using self-attention
Improved generalization compared to CNN models
Smaller gap between training and validation performance
Achieved high validation accuracy of 96.7%
MLP Model

The MLP model is a simple neural network consisting of input, hidden, and output layers. It is one of the most basic deep learning models and uses fully connected layers with ReLU activation, dropout, and BatchNorm1d. The output layer produces class probabilities using softmax.

Confusion Matrix

The confusion matrix shows correct predictions for several classes such as platelets (470), eosinophils (601), and basophils (186). However, some misclassifications occur, particularly between IG and lymphocytes, due to biological similarity.

Classification Report

The MLP model performs well on some classes such as eosinophils and erythroblasts, but shows lower recall for IG. Overall, it achieves an accuracy of 87%, which is lower than CNN and transformer-based models.

Result

The MLP model achieves 87% accuracy. While it performs well on some classes, it struggles with others due to its limited representational power compared to deeper architectures.

Conclusion

The MLP model is simple and easy to implement but is less effective for complex image classification tasks. Advanced models such as CNNs and Vision Transformers significantly outperform it in medical image classification tasks.
