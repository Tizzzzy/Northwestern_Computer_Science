# Recipe Parser Bot for Allrecipes.com

## Introduction
The Recipe Parser Bot is a chatbot designed specifically for [Allrecipes.com](https://www.allrecipes.com). Its primary purpose is to parse recipes provided by users from Allrecipes.com and interact with users through a chat interface, offering detailed recipe information. This project leverages advanced natural language processing techniques to provide a seamless and efficient recipe-fetching experience through simple conversational interactions.

## Installation Guide
To start using the Recipe Parser Bot, follow these steps:

1. **Clone the Repository**: Clone this repository to your local environment.
   ```
   git clone https://github.com/HarryZhao2000/RecipeParserBot.git
   ```
2. **Install Dependencies**: This project depends on Rasa. Ensure that you have Rasa installed. If not, visit [Rasa Official Website](https://rasa.com) for installation instructions. Besides, there are some other required dependencies. Please refer to [requirements.txt](./requirements.txt).
   ```
   pip install -r requirements.txt
   ```

4. **Install en_core_web_sm**: You should download the en_core_web_sm before running our code.
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage Instructions
Follow these steps to run the Recipe Parser Bot:

1. **Run Rasa Actions**:
   ```
   rasa run actions
   ```
   Execute this command in a terminal to start the action server.

2. **Train the Model**:
   ```
   rasa train --force
   ```
   In another terminal, run this command to train the Rasa model.

3. **Start the Chatbot**:
   ```
   rasa shell
   ```
   After the training is complete, run this command to start the chatbot. Please note that the `rasa run actions` should be running while using the chatbot.

## Developing and Testing Environment

This project was developed and tested on an Intel chip Mac running macOS Catalina 10.15 and Python 3.9 running in a virtual environment.

## Connecting to Slack
We have now implemented the functionality to connect the Recipe Parser Bot with Slack. To use this feature, make sure your bot is running in an environment with a public internet URL. Enter this URL and port information into the Interactivity settings of your own Slack App. You can also use [ngrok](https://ngrok.com) to create a temporary public URL for testing in a local environment.

## Example Usage

> **Your input ->  walk me through a recipe from allrecipes.com**
> 
> Sure, please provide the URL.
> 
> **Your input ->  https://www.allrecipes.com/recipe/16311/simple-beef-stroganoff/**
> 
> Got it! The URL is https://www.allrecipes.com/recipe/16311/simple-beef-stroganoff. What do you want to do next?
> 
> 	(1) Show all ingredients
> 
> 	(2) Show basic information
> 
> 	(3) Go to Step 1
> 
> **Your input ->  1**
> 
> Ingredient 1: 1 (8 ounce) package egg noodles
> 
> Ingredient 2: 1 pound ground beef
> 
> Ingredient 3: 1 (10.5 ounce) can fat-free condensed cream of mushroom soup
> 
> Ingredient 4: 1 tablespoon garlic powder
> 
> Ingredient 5: 0.5 cup sour cream
> 
> Ingredient 6: salt and ground black pepper to taste
> 
> **Your input ->  2**
> 
> Time:: 30 minutes
> 
> Prep Time: PT5M
> 
> Cook Time: PT15M
> 
> Total Time: PT20M
> 
> Nutrition: Calories: 679 kcal, Carbon: 48 g, Cholesterol: 159 mg, Fiber: 3 g, Protein: 29 g, Saturated Fat: 17 g, Sodium: 660 mg, Sugar: 2 g, Fat: 41 g, Unsaturated Fat: 0 g
> 
> Rating Value: 4.4
> 
> Rating Count: 918
> 
> Category: ['Dinner']
> 
> Cuisine: ['American']
> 
> **Your input ->  3**
> 
> Step 1:
> 
> Gather all ingredients.
> 
> Ingredients: 1 (8 ounce) package egg noodles; 1 pound ground beef; 1 (10.5 ounce) can fat-free condensed cream of mushroom soup; 1 tablespoon garlic powder; 0.5 cup sour cream; salt and ground black pepper to taste
> 
> Tools: Can opener, Cutting board, Measuring spoons, Measuring cups, Knife, Mixing bowls
> 
> Time: No Time minutes
> 
> **Your input ->  go to step 3**
> 
> Taking you to step 3.
> 
> Step 3:
> 
> Meanwhile, fill a large pot with lightly salted water and bring to a rapid boil. Cook egg noodles at a boil until tender yet firm to the bite, 7 to 9 minutes. Drain and set aside.
> 
> Ingredients: 1 (8 ounce) package egg noodles
> 
> Tools: Colander, strainer, Pot, Timer
> 
> Time: ['7 to 9 minutes'] minutes
> 
> **Your input ->  how to boil water**
> 
> No worries. I found a reference for you: https://www.google.com/search?q=how+to+boil+water
> 
> **Your input ->  idk what is egg noodle**
> 
> No worries. I found a reference for you: https://www.google.com/search?q=what+is+egg+noodle
> 
> **Your input ->  next step**
> 
> Step 4:
> 
> Drain and discard any fat from the cooked beef. Stir condensed soup and garlic powder into the beef. Simmer for 10 minutes, stirring occasionally.
> 
> Ingredients: 1 tablespoon garlic powder, 1 (10.5 ounce) can fat-free condensed cream of mushroom soup, 1 pound ground beef
> 
> Tools: mixing spoon
> 
> Time: ['10 minutes'] minutes
> 
> **Your input ->  what tools do i need**
> 
> Tools: mixing spoon
> 
> **Your input ->  show me all the ingredients**
> 
> Ingredient 1: 1 (8 ounce) package egg noodles
> 
> Ingredient 2: 1 pound ground beef
> 
> Ingredient 3: 1 (10.5 ounce) can fat-free condensed cream of mushroom soup
> 
> Ingredient 4: 1 tablespoon garlic powder
> 
> Ingredient 5: 0.5 cup sour cream
> 
> Ingredient 6: salt and ground black pepper to taste
> 
> Your input ->  how long does this take
> 
> Time: ['10 minutes'] minutes
> 
> **Your input ->  can you transform it vegan**
> 
> OK, it's been transformed to vegan.
> 
> **Your input ->  show me all ingredients**
> 
> Ingredient 1: 1 (8 ounce) package egg noodles
> 
> Ingredient 2: 1 pound crumpled tofu
> 
> Ingredient 3: 1 (10.5 ounce) can fat-free condensed cream of mushroom soup
> 
> Ingredient 4: 1 tablespoon garlic powder
> 
> Ingredient 5: 0.5 cup sour cream
> 
> Ingredient 6: salt and crumpled black pepper to taste
> 
> **Your input ->  can you make it chinese**
> 
> OK, it's been transformed to Chinese.
> 
> **Your input ->  show me all ingredients**
> 
> Ingredient 1: 1 (8 ounce) package egg noodles
> 
> Ingredient 2: 1 pound crumpled tofu
> 
> Ingredient 3: 1 (10.5 ounce) can fat-free condensed cream of mushroom soup
> 
> Ingredient 4: 1 tablespoon garlic powder
> 
> Ingredient 5: 0.5 cup sour cream
> 
> Ingredient 6: salt and crumpled black pepper to taste
> 
> Ingredient 7: soy sauce
> 
> Ingredient 8: shiitake mushrooms
> 
> **Your input ->  i would like the original version**
> 
> This recipe is already the original version.
> 
> **Your input ->  show me all ingredients**
> 
> Ingredient 1: 1 (8 ounce) package egg noodles
> 
> Ingredient 2: 1 pound ground beef
> 
> Ingredient 3: 1 (10.5 ounce) can fat-free condensed cream of mushroom soup
> 
> Ingredient 4: 1 tablespoon garlic powder
> 
> Ingredient 5: 0.5 cup sour cream
> 
> Ingredient 6: salt and ground black pepper to taste
> 
> **Your input ->  bye**
> 
> Refreshed! Please provide a new URL, or say /stop to exit.
> 
> **Your input ->  /stop**

## Contributors
- Ding Zhang
- Dong Shu
- Haoran Zhao
