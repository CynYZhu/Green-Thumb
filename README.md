# Green Thumb AI

Team: Kasha Muzila, Karl Eirich, Robert Turnage

## Introduction 
Welcome to Green Thumb AI, a website and mobile app that can suggest what to plant in a hobby garden based on the number of sun hours, zone, time of year, and user preferences. 

## Live Demo
Please click [here](https://www.youtube.com/watch?v=IWoBIvsvyFo) to see the web demo. 

## Data Engineering
Our database includes over 38,000 flowers, 4,800 herbs, almost 400 vegetables, and over 8,600 fruits to help find all our user's favorite plants they would like to grow. The database was manually curated from multiple web resources to provide ample specificity and sample variety. 

We extracted descriptive features into binary forms, which gives more clear separation of specificity and also allows us to provide additional filters for users. For example, plants that are grains, vegetables, herbs, and fruits are categorized into edibles. Green plants are also given features such as perennial, biennial, or annual for their life cycle. This process improved our information entropy by 43%

## Modeling
Our recommendation engine is designed to first filter out the plants that are not suitable for growth in the users’ geographical environment, which results in a user-specific plant subset. Then it is followed by dimensionality reduction, and use a cosine similarity score to look at the similarity between the plant from the user's shopping cart and the plants in the current pool. The higher the similarity score, the more similarity between the selected item and the recommendations.

Top plants that have the highest similarity scores and not against the user's preferences will be rendered back to the user after each survey. 

Additionally, the recommendation survey in our application allows us to learn user preferences in a set of various plants, giving us the foundation to improve our engine with a user-based filtering approach. 

## Data Privacy
Green Thumb AI is committed to protecting the privacy and accuracy of confidential information to the extent possible, subject to the provision of state and federal law. Our application gives its users the ability to delete their session and all the data that the website collected to help give plant recommendations. 
