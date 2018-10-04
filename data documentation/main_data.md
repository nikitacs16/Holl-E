In this section, keys of the dataset under the data folder have been explained.

This is the collected data organized in dialog. We retain the capitalization, typos if any from the crowdsourced chats.
The dataset is dived into the train, test and dev. Every json file contains a list of datapoints
# Keys and explanation

chat_id: Unqiue key to identify the chats/datapoints. 

chat: The collected chat is stored as list. Every item in the list corresponds to one uterrance. The chat always begins with Speaker1 and Speaker2 follows. Alternate items correspond to respective speakers

documents: These constitute the external knowledge required for the chat. This is a dict and has four keys.
* plot : This is the partial plot of the movie and it is stored as string.
* review : This is one of the picked IMDb review of the movie and it is stored as string.
* comments : This is a list of string where each string is one distinct comment about the movie.
* fact_table: This is again a dict.
  * box_office: The box office collection of the movie. Stored as a string
  * awards : This is a list of (atmost) five awards/nominations of the movie. Stored as a list of strings
  * taglines : This is a list of (atmost) five taglines of the movie. Stored as a list of strings
  * similar_movies : This is a list of atmost five movies similar to the movie. Stored as a list of strings

imdb_id: IMDb id about the movie on which the chat is based. This is a string.

labels: This is the list of numbers associated with every utterance from the dialog. The integer indicates which resource is chosen for producing the response. 0 is for plot, 1 is for review, 2 is for comments, 3 is for fact_table and 4 is for none i.e, no resource is selected

movie_name: The movie name on which the chat is based. Stored as a string

spans: The contigous set of words selected from a document to formulate a response. This is a list of strings. In case of no span, an empty string is used.
