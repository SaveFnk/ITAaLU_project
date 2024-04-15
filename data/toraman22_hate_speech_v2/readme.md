# Toraman22 Hate Speech Dataset v2

We acknowledge that some annotations in the original dataset (v1) are controversial. Therefore, we publish a more reliable dataset version (v2) that includes only the tweets with more than 80% annotator agreement. The dataset v2 has 128,907 tweets. 60,310 of them are Turkish, and 68,597 are English. Explanations of the columns of the file are as follows:

tweet_id
user_id	
user_name	
screen_name
verified: If user's profile is verified.	
created_at: The date that user's profile is created.	
friends_count: User's total number of followees	
followers_count: User's total number of followers	
statuses_count: User's total number of sharings	
favourites_count: User's total number of likes	
default_piclabel: User's profile picture	
topic: Religion (0), Gender (1), Race (2), Politics (3), or Sports (4)
language: Turkish (0) or English (1)	
date: Tweet's published date	
text: Tweet's text contents
label_0: How many times tweet is labeled Normal (0)
label_1: How many times tweet is labeled Offensive (1)
label_2: How many times tweet is labeled Hate (2)	
label_score: Final annotation label

Please cite the following paper.

@InProceedings{toraman2022large,
  author    = {Toraman, Cagri  and  \c{S}ahinu\c{c}, Furkan and Yilmaz, Eyup Halit},
  title     = {Large-Scale Hate Speech Detection with Cross-Domain Transfer},
  booktitle = {Proceedings of the Language Resources and Evaluation Conference},
  month     = {June},
  year      = {2022},
  address   = {Marseille, France},
  publisher = {European Language Resources Association},
  pages     = {2215--2225},
  url       = {https://aclanthology.org/2022.lrec-1.238}
}
