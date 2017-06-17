# Meetup.com Group Category Predictions
The goal is to predict Meetup.com group category based on event description.

From a human interest perspective, this goal speaks to how we communicate and how that communication can be condensed and categorized. From a business perspective, being able to predict a group category based on event description could be useful for auto-suggesting tags for events. A feature like this would help users discover events they interested in more easily.

From a technical perspective, as this is a NLP problem, I make use of the NLTK package as well as scikit-learn's CountVectorizer and TfidfVectorizer feature extraction tools. In terms of model selection, I employ both Random Forest and Naive Bayes as both have been show to perform well with text data.
