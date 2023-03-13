# Naive_Bayes_Understanding_the_Algorithm

## Description
This project aims to understand how the Naive-Bayes algorithm works in classification models. For this, an analysis was performed using the algorithm, as well as some research, to understand what steps the algorithm takes to establish the correct classification.

## Used Technologies
- **Python** - The source code is written in Python
- **Jupyter Notebook** - The platform of choice to implement this project
- **Python libraries** - Numpy, Pandas, Matplotlib, Scikit Learn, Statistics, Imblearn, Seaborn

## The algorithm
### How it works
The algorithm's logic is based on probability concepts. According to the Scikit-Learn documentation, the Naive-Bayes methods are actually a set of supervised learning algorithms. The name comes from the fact that the algorithm applies Bayes' theorem naively assuming that each pair of attributes has a conditional independence between its elements. Here, the method used is the Gaussian Naive-Bayes. The gereralized formula has a lot of parameters, but in its simplest form, it can be written as follows:

<p align="center"><img width="400px" src="https://user-images.githubusercontent.com/46689116/224799887-332bbe80-e7ef-457d-bf62-8963869f1edc.png"/></p>

So, to determine the probability of **y** (the target or hypothesis) occuring based on the event of the observation **X1...Xn**, we have to determine the probability of **y** happening regardless anything else, the probability of **X1...Xn**, e.g., the attributes with certain values, occuring based on the event of the observation **y** and, finally, the probability of **X1...Xn**. It seems a bit confusing at first, but it's actually quite simple.

As the part **P(X1...Xn)** corresponds to a number of observations independent of the hypothesis, it can be omitted because its values would always be constant. So we have:

<p align="center"><img width="400px" src="https://user-images.githubusercontent.com/46689116/224809247-8a864a36-d6d3-47d3-a0ef-2e5f148f86b1.png"/></p>

The **P(y)** can be calculated taking, for every value of **y**, the number of occurencies of this value in the dataset. So, if we have five samples in which y = [1,1,0,0,1], the P(1) = 3/5 and the P(0) = 2/5.

The **P(X1...Xn|y)** can be calculated taking, for every value of **y**, the number of occurencies of each value of the attributes in X. So, if we have Gender as an attributes with the possible values of F and M, and we take five samples like X['Gender'] = [F,M,F,F,M], assosciated with the five samples of **y** we had before, we have to calculate P(F|1), P(M|1), P(F|0) and P(M|0). For P(F|1), we have &rarr; P(F|1) = 1/3, as we only have one occurency of the Gender being 'F' while the target being 1, which occurs three times.

We all the calculus made, we can determine the P(y|X1...Xn). Here, to determine P(1|M), for example, we have: 

P(1|M) = P(1)×P(M|1) &rarr; P(1|M) = 3/5×2/3 &rarr; P(1|M) = 0.4 

So there's a chance of 40% that y =1 if X['Gender'] = M.

For knowledge (in this example):
- P(1|F) = 0.2
- P(0|M) = 0
- P(0|F) = 0.4

So, in this scenario, if a new input was X['Gender'] = 'F', the probability of y = 1 would be 20% and the probability of y = 0 would be 40%.
Of course it would take a lot of other attributes to be able to classify new data correctly, but the logic is always the same. The probability calculated for each attribute associated with a target has to be combined to get to a real probability of the target. 

### The used dataset
The dataset used to perform the analysis is called 'BankChurners'. It has data on bank customers who did or did not churn and their characteristics, such as age, gender, credit limit, card category, etc. It is important to say that I took the dataset from the Kaggle site and some attributes were related to an analysis performed with Naive-Bayes. These columns were eliminated to prevent the model from being biased. Furthermore, the dataset was nearly perfect to be classified with the algorithm, as only some of the attributes were highly correlated with each other.

## Steps followed
To build this project, a few steps were followed:
1. A few functions were defined to ease the coding and keep the code clean. Such as:
    - A function to adjust the attributes, turning the non-numerical ones into numerical
    - A function to split the dataset using different inputs for X and y 
    - And, a function to train the model and make the predictions with different Train an Test sets
2. Then, the unnecessary columns to the analysis were dropped
    - To do this, a heatmap were constructed based on the dataset
    - The highly correlated attributes (with more than 0.4 of correlation indication) were analyzed
    - And, some of them were dropped due to the nature of the classifier, which assumes that the attributes are independent of each other
3. The dataset was analyzed to verify if there were any missing values (and, in a positive scenario, fill or discard them)
4. After training the model with the raw data and making the predictions, the data was balanced to analyze how this would affect the evaluation metrics
5. The data was then normalized for the same reason
6. The evaluation metrics obtained from each model (trained with different sets) were compared

## Preview of the analysis
The information of the dataset before any treatment:

![df_before](https://user-images.githubusercontent.com/46689116/224067280-6da2b2f7-84db-4ac7-9477-3e9540d44b20.png)

The information of the dataset after the non-numerical attributes were encoded and some of the highly correlated attributes were dropped:

![df_after](https://user-images.githubusercontent.com/46689116/224067568-61c96108-276f-46c6-830f-dc9f12624c66.png)

The heatmap generated based on the dataset, that shows how much the attributes are correlated:
![heatmap_nb](https://user-images.githubusercontent.com/46689116/224124621-efede896-9551-4d5f-a00a-e20b856bda57.png)

With the map, we can see that the attributes 'Credit_Limit' and 'Avg_Open_To_Buy' are highly corretaled, as well as the 'Months_on_book' and 'Customer_Age', among others. An example of a pair of attributes that are not correlated is 'Months_on_book' and 'Marital Status'.

To make this clearer, I plotted all of the examples, to see how the data was distributed.
Here we have:
- 'Credit_Limit' vs. 'Avg_Open_To_Buy'

![relation_btw_correlated_attributes](https://user-images.githubusercontent.com/46689116/224065039-7a767bbb-75c1-4c54-985b-9c001ceb4ee3.png)

- 'Months_on_book' vs. 'Customer_Age'

![relation_btw_correlated_attributes_2](https://user-images.githubusercontent.com/46689116/224065074-2549c22d-5062-4e9d-b85a-022bd38c2c5d.png)

We can see that in both cases the attributes have values that are directly proportional. When the values on the first one increases, the other follows the tendency, also increasing.

- 'Months_on_book' vs. 'Marital Status'

![relation_btw_uncorrelated_attributes](https://user-images.githubusercontent.com/46689116/224064963-535a18b8-1230-4a87-9dbe-5ba96a5c7169.png)

Here we see that the attributes are not related at all. 
First the model training and the predictions were done with the raw data, and the evalution metrics results were as follows:

![cr_raw_data](https://user-images.githubusercontent.com/46689116/224124323-2a75dd01-affa-458c-bd7a-448343e038fb.png)

We can see that the Accuracy was high enough, on the other hand, we don't have values for Precision and F1-score, as both are zero. 
After a little research, I learned that are some reasons for this to happen. Among them are: 
- Some labels predicted among the set with true labels, y_true, are not in the predictions, y_pred
- There is a set between the train and test sets that is significantly smaller than the other
- And, according to the Sckit-Learn documentatio, **Precision** is undefined when **True Positive** + **False Positive** == 0, and **Recall** is undefined when **True Positive** + **False Negative** == 0. By default, the metric will be set to 0, as will F1-Score.
The case here is the last one, as we can see the Confusion Matrix generated for the model:

![cm_w_warning](https://user-images.githubusercontent.com/46689116/224149910-0b6c8a0c-c28e-4fff-8ff1-418185eaab5a.png)

The Scikit-Learn documentation suggests that this behavior can be modified with zero_division parameter. As I wanted to improve the evaluation metrics, I decided to try and balance the samples, to see if the predictions would have this metrics defined and if the other ones would be improved.

I tried two methods: Oversampling and Undersampling.
- The classification report for the Oversampling method was as follows:
![cr_oversampled_data](https://user-images.githubusercontent.com/46689116/224124943-0971732b-2cdc-4b34-8d25-3e930d892e7c.png)

- The classification report for the Undersampling method was as follows:
![cr_undersampled_data](https://user-images.githubusercontent.com/46689116/224123295-a7cc1310-cab8-4cc0-8cf3-2be486eb707c.png)

Both of them eliminated the warning raised with the condition described above, but the accuracy has decreased significantly.

So, as our data is now numerical, but in different scales, I decided to try and normalized the raw data, to improve the model performance and possibly increase the values of our evaluation metrics. The result was as follows:

![cr_normalized_data](https://user-images.githubusercontent.com/46689116/224123861-4303eb66-6f5a-4fc1-9799-24c4987eb333.png)

As we can see, all of the evaluation metrics had their values increased and both the **Precision** and **F1-Score** has now values different from zero.
As expected, we don't have the **True Positive** + **False Positive** == 0 problem anymore, as the confusion matrix below shows:

![cm_wo_warning](https://user-images.githubusercontent.com/46689116/224127877-01921de6-1b74-411a-acfc-ec552135677a.png)


## Conclusion 
The built-in algorithm of the Gaussian Naive-Bayes it's pretty simple to use. Although there is room for improvement, the accuracy achieved, as well as the other metric evaluations, had satisfatory results. Some considerations about the analysis are: 
- Balancing the data did not improved the evalution metrics values. And, it is important to say that this is a pretty good example that this doesn't always work. 
    - Balance data can create a definite bias in predictions, impacting the consistency of the results as the sample size grows. 
    - Balanced data can make the analysis loose information, as the frequencies of the attributes change, affecting production performance.
    - The truth is, if the training set is large enough to guarantee the confiability of an analysis, artificial balancing is rarely useful or necessary. 
    - To sum it up, in some cases, balanced data can be worse than unbalanced data.

About the algorithm:
- The attributes had to be treated so that the non-numerical attributes would assume their numerical equivalencies, to be treated by the algorithm.
- The naive assumption that the attributes has a conditional independence between them can impact the analysis, as this is not always true. Yet, the classification was very satisfactory.
- Once the sample was normalized, the behavior of the algorithm with medium size samples proved to be satisfactory, as our dataset was not very large.

In short, this is a good classifier, but with room for improvement, like all other algorithms, the right use case can be decisive in determining the performance of the algorithm. Scikit-Learn itself suggests that it can be used when the analysis has less than 100k samples and the data is text data. But the **right** algorithm depends on the data you have and how well it's been pre-processed (with cleaning, padding and encoding).
