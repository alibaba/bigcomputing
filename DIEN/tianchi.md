# Introduction
## Dataset
We random select about 8 million users with over 15M shopping samples. Each sample consists of 100 user history data. The dataset is organized in a very similar form to MovieLens-20M, i.e., each line represents a specific user-item interaction and history interactions, including key information like user ID, item ID, category ID, shop ID, node ID, product ID, brand ID. The dataset is split into training and testing. Training has 15M samples and testing has 0.97M samples.

Dimensions of the dataset are

| Dimension            | Dimension Size | Feature Exaplanation|
|----------------------|----------------|---------------------|
| Number of users      | 7956430        |An integer that represents a user|
| Number of items      | 34196611       |An integer that represents an item|
| Number of categories | 5596           |An integer that represents the category which the corresponding item belongs to|
| Number of shops      | 4377722        |An integer that represents a shop |
| Number of nodes      | 2975349        |An integer that represents a cluster which some items belong to|
| Number of products   | 65624          |An integer that represents a product|
| Number of brands     | 584181         |An integer that represents a brand|
| Number of interactions| 15M          |An integer that represents a sample|

**Citations**
Guorui Zhou,* Na Mou,† Ying Fan, Qi Pi, Weijie Bian,. Chang Zhou, Xiaoqiang Zhu, Kun Gai. Deep Interest Evolution Network for Click-Through Rate Prediction. In Proceedings of the 28th _AAAI_ Conference on Artificial Intelligence, 1369–1375.