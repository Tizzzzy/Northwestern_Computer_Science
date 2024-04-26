from urllib.request import urlopen
import re
import pandas as pd
import unicodedata
import requests
from bs4 import BeautifulSoup
import json
import spacy
import nltk
import string
import random
import copy

action_tool_lexicon = {'whip': ['wire whisk', 'rotary beater'],
                       'scald': ['container'], 'grate': ['grater'],
                       'beat': ['mixing spoon', 'wire whisk', 'rotary beater', 'electric mixer', 'Mixing bowl'],
                       'chop': ['knife', 'food chopper', 'Cutting board'],
                       'combine': ['mixing spoon', 'wire whisk'],
                       'dice': ['knife'], 'cube': ['knife'],
                       'pare': ['vegetable peeler'],
                       'stir': ['mixing spoon'],
                       'fold in': ['mixing spoon', 'rubber scrapper'],
                       'blend': ['mixing spoon', 'wire whisk', 'rotary beater', 'electric mixer'],
                       'blanch': ['pot'], 'cream': ['rotary beater', 'mixing spoon'],
                       'sift': ['flour sifter', 'strainer'],
                       'shred': ['Grater', 'Food processor'],
                       'knead': ['Work surface', 'Hands'],
                       'baste': ['baster', 'brush'],
                       'mix': ['mixing spoon', 'wire whisk', 'rotary beater', 'electric mixer'],
                       'mince': ['knife', 'scissors', 'Cutting board'],
                       'puree': ['Blender', 'Food processor'],
                       'marinate': ['Marinating dish', 'zip-top bag', 'Refrigerator'],
                       'score': ['knife'],
                       'dilute': ['Water', 'Measuring cup', 'Stirring spoon'],
                       'gather': ['Mixing bowls', 'Measuring cups', 'Measuring spoons', 'Cutting board', 'Knife', 'Can opener'],
                       'preheat': ['Oven', 'Oven thermometer'],
                       'grease': ['Baking dish', 'Cooking spray', 'Brush', 'paper towel'],
                       'sauté': ['Sauté pan', 'Cooking oil', 'butter', 'Spatula'],
                       'brown': ['Skillet', 'frying pan', 'Spatula'],
                       'boil': ['Pot', 'Water', 'Stovetop'],
                       'cook': ['Pot', 'Colander', 'strainer', 'Timer'],
                       'grill': ['Grill', 'Tongs', 'Marinade brush'],
                       'bake': ['Oven', 'Baking sheet', 'dish', 'Timer'],
                       'simmer': ['Saucepan', 'Spoon', 'whisk'],
                       'stir-fry': ['Wok', 'skillet', 'Cooking oil', 'Spatula'],
                       'whisk': ['Whisk', 'Bowl'],
                       'fry': ['Frying pan', 'Cooking oil', 'butter', 'Spatula'],
                       'season': ['Salt shaker', 'Pepper mill'],
                       'garnish': ['Herb scissors', 'knife', 'Cutting board'],
                       'serve': ['Plates', 'bowls', 'Utensils']
                       }

transform_from_vegan_steps = {
    "1 slice of bacon": ["put the bacon in a non-stick frying pan set over medium heat, cook for 3 minutes on each side, let it sit and cut into pieces",
                          "add the chopped bacon pieces onto the dish, dish up!"]
}


unhealthy_oil = "100 grams of lard"


southeast_asian_ingredients = {
    "sauce" : ["fish sauce", "oyster sauce", "sambal", "curry paste", "nam phrik"],
    "ingredients": ["shrimp paste", "pandan leaves", "chilli peppers", "lemongrass stalk", "turmeric"]
}

chinese_ingredients = {
    "sauce" : ["soy sauce", "oyster sauce", "cooking wine", "dark soy sauce", "seasame oil", "black vinegar", "chili oil"],
    "ingredients" : ["scallion", "ginger", "garlic", "dried chili peppers", "five spice powder", "Sichuan peppercorns", "shiitake mushrooms"]
}

cooking_methods = ["stew", "steam", "poach", "bake", "blanch", "deep fry", "braise", "roast", "simmer", "stir", "stir fry", "grill", "sauté", "boil", "fry"]

class Ingredient_List_Builder:

    def __init__(self, url):
        self.general_foods = pd.read_csv('generic-food.csv')
        self.url = url

        self.vegetable_list = []
        self.meat_list = []
        self.meat_sub = 'tofu'
        self.dairy_list = []
        self.build_vegetable_list()
        self.build_meat_list()

    def build_vegetable_list(self):
        vegetable = self.general_foods[self.general_foods['GROUP'] == 'Vegetables']
        self.vegetable_list = vegetable["FOOD NAME"].to_list()
        self.vegetable_list = [v.lower() for v in self.vegetable_list]


    def build_meat_list(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 ...'
        }

        response = requests.get(self.url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            meat_products = soup.find_all('ul')

            for list in meat_products:
                for item in list.find_all('li'):
                    for a in item.find_all('a'):
                        a.decompose()
                    text = item.get_text(strip=True)
                    if text:
                        split_text = [t.strip() for t in text.split(',')]
                        self.meat_list.extend(split_text)
        else:
            print(f"Failed to retrieve the webpage for meat list! Status code: {response.status_code}")

        meat = self.general_foods[self.general_foods['GROUP'] == 'Animal foods']
        meat_list = [m.lower() for m in meat["FOOD NAME"].to_list() if m.lower() not in self.meat_list]
        self.meat_list.extend(meat_list)
        self.meat_list = [m.lower() for m in self.meat_list]


class Step:
    def __init__(self, action="", methods=[], ingredients=[], tools=[], time="", temperature=""):
        self.action = action
        self.methods = methods
        self.ingredients = ingredients
        self.tools = tools
        self.time = time


class Recipe:
    def __init__(self, link):
        # actions + ingredients stored here should not be changed.
        # when we perform transformations, we should modify "Step()"
        # stored in self.steps.
        self.link = link
        self.name = ''
        self.ingredients = []
        self.ingredients_step = []
        self.actions = []
        self.steps = []
        self.utensils = []
        self.tools = []
        self.other_parameters = []
        self.stop_words = ['brown', 'done']
        self.meat_url = 'https://naturalhealthtechniques.com/list-of-meats-and-poultry/'

        # for transforms:
        self.general_foods = pd.read_csv('generic-food.csv')
        self.ingredient_list_builder = Ingredient_List_Builder(self.meat_url)

        # a deep copy of self.steps. if we make transformations, we make edits
        # in steps_copy, not the original steps.
        self.steps_copy = []

        # pointer to the original recipe
        self.org_recipe = None

    def load(self):
        global nlp
        nlp = spacy.load("en_core_web_sm")
        nltk.download('popular')
        nltk.download('stopwords')
        html = self.get_recipe()
        self.parse(html)
        self.get_step()
        self.steps_copy = copy.deepcopy(self.steps)


    def get_recipe(self):
        """
        link: str
        get HTML and extract
        """
        try:
            page = urlopen(self.link)
            html_bytes = page.read()
            html = html_bytes.decode("utf-8")
            return html
        except:
            assert("Failed to load recipe!")


    def parse(self, html):
        """
        Input: instructions (str)
        Return: step object
        parse instructions and extract information to a step object
        """

        soup = BeautifulSoup(html, 'html.parser')

        # Find the recipeInstructions section
        recipe_instructions = soup.find("script", {"type": "application/ld+json"})
        if recipe_instructions:
            recipe_instructions = json.loads(recipe_instructions.text)
            print(recipe_instructions)

        recipe_instructions = recipe_instructions[0]

        name = ''
        ingredients = []
        action = []
        tools = []
        utensils = []
        other_parameters = []

        if 'name' in recipe_instructions:
            name = recipe_instructions['name']

        if 'recipeIngredient' in recipe_instructions:
            for ingre in recipe_instructions['recipeIngredient']:
                ingredients.append(ingre)

        if 'recipeInstructions' in recipe_instructions:
            for step in recipe_instructions["recipeInstructions"]:
                if step.get("@type") == "HowToStep" and "text" in step:
                    action.append(step["text"])

        if 'description' in recipe_instructions:
            print(recipe_instructions['description'])
            time_match = re.search(r'\b\d+\s*\w+\b', recipe_instructions['description'])
            if time_match:
                time = time_match.group()
                other_parameters.append({"Time:": time})
            else:
                other_parameters.append({"Time:": "Time not found in the sentence."})

        if 'prepTime' in recipe_instructions:
            other_parameters.append({"Prep Time": recipe_instructions['prepTime']})

        if 'cookTime' in recipe_instructions:
            other_parameters.append({"Cook Time": recipe_instructions['cookTime']})

        if 'totalTime' in recipe_instructions:
            other_parameters.append({"Total Time": recipe_instructions['totalTime']})

        if 'nutrition' in recipe_instructions:
            nutrition = recipe_instructions['nutrition']
            if 'calories' in nutrition:
                calories = nutrition['calories']
            else:
                calories = '0 kcal'
            if 'carbohydrateContent' in nutrition:
                carbon = nutrition['carbohydrateContent']
            else:
                carbon = '0 g'
            if 'cholesterolContent' in nutrition:
                cholesterol = nutrition['cholesterolContent']
            else:
                cholesterol = '0 mg'
            if 'fiberContent' in nutrition:
                fiber = nutrition['fiberContent']
            else:
                fiber = '0 g'
            if 'proteinContent' in nutrition:
                protein = nutrition['proteinContent']
            else:
                protein = '0 g'
            if 'saturatedFatContent' in nutrition:
                saturatedFat = nutrition['saturatedFatContent']
            else:
                saturatedFat = '0 g'
            if 'sodiumContent' in nutrition:
                sodium = nutrition['sodiumContent']
            else:
                sodium = '0 g'
            if 'sugarContent' in nutrition:
                sugar = nutrition['sugarContent']
            else:
                sugar = '0 g'
            if 'fatContent' in nutrition:
                fat = nutrition['fatContent']
            else:
                fat = '0 g'
            if 'unsaturatedFatContent' in nutrition:
                unsaturatedFat = nutrition['unsaturatedFatContent']
            else:
                unsaturatedFat = '0 g'
            nutrition_value = f"Calories: {calories}, Carbon: {carbon}, Cholesterol: {cholesterol}, Fiber: {fiber}, Protein: {protein}, Saturated Fat: {saturatedFat}, Sodium: {sodium}, Sugar: {sugar}, Fat: {fat}, Unsaturated Fat: {unsaturatedFat}"
            other_parameters.append({'Nutrition': nutrition_value})

        if 'aggregateRating' in recipe_instructions:
            rating = recipe_instructions['aggregateRating']
            if 'ratingValue' in rating:
                ratingValue = rating['ratingValue']
                other_parameters.append({"Rating Value": ratingValue})
            if 'ratingCount' in rating:
                ratingCount = rating['ratingCount']
                other_parameters.append({"Rating Count": ratingCount})


        if 'recipeCategory' in recipe_instructions:
            other_parameters.append({"Category": recipe_instructions["recipeCategory"]})

        if 'recipeCuisine' in recipe_instructions:
            other_parameters.append({"Cuisine": recipe_instructions["recipeCuisine"]})


        # print('name: ', name)
        # print('ingredients: ', ingredients)
        # print('action: ', action)
        # print('other_parameters: ', other_parameters)

        self.name = name
        self.ingredients = ingredients
        self.actions = action
        self.other_parameters = other_parameters

    def get_step(self):
        """
        parse instructions and get steps
        """
        num_steps = len(self.actions)
        cook_verbs = self.get_action_verbs()
        self.tools = self.get_tool(cook_verbs)
        self.extract_ingredients()
        for i in range(num_steps):
            assert len(self.ingredients_step) == len(self.tools) == len(self.actions), "Length should be equal to each other!"
            time = self.extract_time(self.actions[i])
            if not time:
                time = "No Time"
            step = Step(self.actions[i], cook_verbs[i], self.ingredients_step[i], self.tools[i], time)
            self.steps.append(step)

    """
    Below are all helper functions
    """

    def get_action_verbs(self):
        """
        Helper Function to get all the verbs within the action list
        """
        total = len(self.actions)
        cook_actions = []
        for step in range(total):
            # Process the text using spaCy
            doc = nlp(self.actions[step])

            #print(doc[0].lemma_, doc[0].pos_)
            # Extract the verbs from the processed text
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB" and token.lemma_ not in self.stop_words]

            # Delete this if needed
            if doc[0].pos_ != "ADV" and doc[0].lemma_ not in verbs:
                verbs.insert(0, doc[0].lemma_)
            # Print the verbs for each step
            cook_actions.append(verbs)
        return cook_actions


    def get_tool(self, action_words):
        tools = []
        for words in action_words:
            temp = []
            for word in words:
                if word.lower() in action_tool_lexicon:
                    temp.extend(action_tool_lexicon[word.lower()])
            tools.append(list(set(temp)))
        return tools


    def remove_numbers_stopwords_punctuation(self, tokens):
        """
        Helper Function to remove stopwords and punctuations to help
        extract required ingredients from each step
        """
        # Remove numbers
        tokens = [token for token in tokens if not token.isdigit()]

        # Remove common English stopwords
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]

        tokens = [token for token in tokens if token not in string.punctuation]

        return tokens


    def extract_ingredients(self):
        """
        Extract ingredients from each step
        """
        total = len(self.actions)
        for step in range(total):
            ingredient = set()
            if 'all ingredient' in self.actions[step]:
                #print(f"step{step+1} involves ingredient: {self.ingredients}")
                self.ingredients_step.append(self.ingredients)

            else:
                tokens = nltk.tokenize.word_tokenize(self.actions[step])
                tokens = self.remove_numbers_stopwords_punctuation(tokens)
                for token in tokens:
                    for ingred in self.ingredients:
                        if token in ingred:
                            ingredient.add(ingred)
                if ingredient:
                    self.ingredients_step.append(ingredient)
                else:
                    self.ingredients_step.append(["No ingredients needed"])


    def extract_time(self, text):
        # Define a regular expression pattern for matching time expressions including minutes, hours, and seconds
        time_pattern = re.compile(r'\b(\d+)\s*(?:to|-)?\s*(\d+)?\s*(?:(minutes?|mins?)|(hours?|hrs?)|(seconds?|secs?))\b', re.IGNORECASE)

        # Find all matches in the given text
        matches = re.findall(time_pattern, text)

        # Extract and format the matched time information
        result = []
        for match in matches:
            start_time, end_time, unit_minutes, unit_hours, unit_seconds = match

            if unit_minutes:
                time_str = f'{start_time} to {end_time} minutes' if end_time else f'{start_time} minutes'
            elif unit_hours:
                time_str = f'{start_time} to {end_time} hours' if end_time else f'{start_time} hours'
            elif unit_seconds:
                time_str = f'{start_time} to {end_time} seconds' if end_time else f'{start_time} seconds'
            else:
                continue

            result.append(time_str)

        return result

    """
    Transformations below
    """

    def transform_to_vegan(self, org_recipe):
        self.steps_copy = copy.deepcopy(self.steps)
        meat_sub = self.ingredient_list_builder.meat_sub
        meat_list = self.ingredient_list_builder.meat_list
        # replace meat products with meat_substitue (tofu)
        for step in self.steps_copy:
            if "ground" in step.action:
                step.action = step.action.replace("ground", "crumpled")
            if "fat" in step.action:
                step.action = step.action.replace("fat", "okara")
            for meat in meat_list:
                step.action = step.action.replace(meat, meat_sub)

            for i in range(len(step.ingredients)):
                step.ingredients = list(step.ingredients)
                for meat in meat_list:
                    step.ingredients[i] = step.ingredients[i].replace(meat, meat_sub)
                    step.ingredients[i] = step.ingredients[i].replace("ground", "crumpled")
            print("ingredients: ", step.ingredients)
            print("action: ", step.action)

        # create a new transformed recipe
        transformed_recipe = copy.deepcopy(org_recipe)
        transformed_recipe.steps = self.steps_copy
        transformed_recipe.org_recipe = org_recipe

        for meat in meat_list:
            for i in range(len(transformed_recipe.ingredients)):
                transformed_recipe.ingredients[i] = transformed_recipe.ingredients[i].replace(meat, meat_sub)
                transformed_recipe.ingredients[i] = transformed_recipe.ingredients[i].replace("ground", "crumpled")
        return transformed_recipe


    def transform_from_vegan(self, org_recipe):
        """
        Transform from a given vegetarian recipe to non-vegetarian
        """
        self.steps_copy = copy.deepcopy(self.steps)
        protein = False
        meat_list = self.ingredient_list_builder.meat_list
        meat_sub = self.ingredient_list_builder.meat_sub

        # create a new transformed recipe
        transformed_recipe = copy.deepcopy(org_recipe)
        transformed_recipe.steps = self.steps_copy
        transformed_recipe.org_recipe = org_recipe

        for step in self.steps_copy:
            for meat in meat_list:
                if meat in step.action:
                    print("The given recipe is not vegetarian!\n")
                    return
            if meat_sub in step.action:
                protein = True
                break
        transformed_recipe.ingredients = list(transformed_recipe.ingredients)
        if protein:
            meat_choice = random.choice(meat_list)
            for step in self.steps_copy:
                step.action = step.action.replace(meat_sub, meat_choice)
                for i in range(len(step.ingredients)):
                    step.ingredients = list(step.ingredients)
                    step.ingredients[i] = step.ingredients[i].replace(meat_sub, meat_choice)
                print("ingredients: ", step.ingredients)
                print("action: ", step.action)
            for i in range(len(transformed_recipe.ingredients)):
                transformed_recipe.ingredients[i] = transformed_recipe.ingredients[i].replace(meat_sub, meat_choice)

        else:
            for key, value in transform_from_vegan_steps.items():
                added_meat = key
                added_steps = value
            step1 = Step(added_steps[0], [added_meat])
            step2 = Step(added_steps[1], [added_meat])

            self.steps_copy.insert(1, step1)
            # acquire the last step in the step_copy list
            step = self.steps_copy[-1]
            if step.ingredients[0] != 'No ingredients needed':
                self.steps_copy.append(step2)
            else:
                self.steps_copy.insert(-1, step2)

            for step in self.steps_copy:
                print("ingredients: ", step.ingredients)
                print("action: ", step.action)
            transformed_recipe.ingredients.append(added_meat)
        return transformed_recipe


    def transform_to_unhealthy(self, org_recipe):
        """
        TODO: search for actions. replace fry / steam ... to deep fry
        - First check cooking methods. See if it can be deep fried (method in same cooking_method key)
        - If cannot be deep fried ()
        """
        self.steps_copy = copy.deepcopy(self.steps)
        oil_org = None

        # create a new transformed recipe
        transformed_recipe = copy.deepcopy(org_recipe)
        transformed_recipe.steps = self.steps_copy
        transformed_recipe.org_recipe = org_recipe

        transformed_recipe.ingredients = list(transformed_recipe.ingredients)
        transformed_recipe.ingredients.append(unhealthy_oil)

        for j in range(len(self.steps_copy)):
            # change regular oil to lard to make it unhealthy
            self.steps_copy[j].ingredients = list(self.steps_copy[j].ingredients)
            for i in range(len(self.steps_copy[j].ingredients)):
                if "oil" in self.steps_copy[j].ingredients[i]:
                    oil_org = self.steps_copy[j].ingredients[i].split()
                    oil_org = oil_org[-2:]
                    self.steps_copy[j].ingredients[i] = unhealthy_oil
            if type(oil_org) == list:
                oil_org = ' '.join(oil_org)
            # if there is oil mentioned in the recipe
            if oil_org != None:
                self.steps_copy[j].action = self.steps_copy[j].action.replace(oil_org, unhealthy_oil)
            else:
                action_verbs = self.get_action_verbs()
                i = 0
                flag = False
                detected_verb = None
                for i in range(len(action_verbs)):
                    for verb in action_verbs[i]:
                        if verb.lower() in cooking_methods:
                            flag = True
                            detected_verb = verb
                    if flag:
                        break
                if flag and (j == i):
                    self.steps_copy[j].ingredients.append(unhealthy_oil)
                    self.steps_copy[j].action = "Add " + unhealthy_oil + " into the pan. " + self.steps_copy[j].action
                    self.steps_copy[j].action = self.steps_copy[j].action.replace(detected_verb, "Heat the " + unhealthy_oil + " until melted. Mix and " + detected_verb)
            print("ingredients: ", self.steps_copy[j].ingredients)
            print("action: ", self.steps_copy[j].action)
        return transformed_recipe


    def transform_to_southeast_asian(self, org_recipe):
        self.steps_copy = copy.deepcopy(self.steps)
        sea_sauce = southeast_asian_ingredients["sauce"]
        sea_ing = southeast_asian_ingredients["ingredients"]

        # first, add some special sauces to change the cusine
        sauce = random.choice(sea_sauce)
        step1 = Step("add one tablespoon of " + sauce + " on top of the dish, dish up!", "one tablespoon of " + sauce)
        step = self.steps_copy[-1]

        # create a new transformed recipe
        transformed_recipe = copy.deepcopy(org_recipe)
        transformed_recipe.steps = self.steps_copy
        transformed_recipe.org_recipe = org_recipe
        transformed_recipe.ingredients = list(transformed_recipe.ingredients)
        transformed_recipe.ingredients.append(sauce)

        step.ingredients = list(step.ingredients)
        if step.ingredients[0] != "No ingredients needed":
            self.steps_copy.append(step1)
        else:
            self.steps_copy.insert(-1, step1)

        # second, add some particular ingredients that could represent the transformed cusine
        ing = random.choice(sea_ing)

        action_verbs = self.get_action_verbs()
        i = 0
        # check if certain cooking methods exist in the steps. If there are,
        # then add the ingredients into that step
        flag = False
        for i in range(len(action_verbs)):
            for verb in action_verbs[i]:
                if verb.lower() in cooking_methods:
                    flag = True
            if flag:
                break
        if flag:
            step_ing = self.steps_copy[i]
            step_ing.ingredients = list(step_ing.ingredients)
            step_ing.ingredients.append("20g of " + ing)
            step_ing.action = "Add chopped 20g of " + ing + " and mix with other ingredients. " + step_ing.action
            transformed_recipe.ingredients.append(ing)

        for step in self.steps_copy:
            print("ingredients: ", step.ingredients)
            print("action: ", step.action)

        return transformed_recipe


    def transform_to_Chinese(self, org_recipe):
        self.steps_copy = copy.deepcopy(self.steps)
        chn_sauce = chinese_ingredients["sauce"]
        chn_ing = chinese_ingredients["ingredients"]

        # first, add some special sauces to change the cusine
        sauce = random.choice(chn_sauce)
        step1 = Step("add one tablespoon of " + sauce + " on top of the dish, dish up!", "one tablespoon of " + sauce)
        step = self.steps_copy[-1]

        # create a new transformed recipe
        transformed_recipe = copy.deepcopy(org_recipe)
        transformed_recipe.steps = self.steps_copy
        transformed_recipe.org_recipe = org_recipe
        transformed_recipe.ingredients = list(transformed_recipe.ingredients)
        transformed_recipe.ingredients.append(sauce)

        step.ingredients = list(step.ingredients)
        if step.ingredients[0] != "No ingredients needed":
            self.steps_copy.append(step1)
        else:
            self.steps_copy.insert(-1, step1)

        # second, add some particular ingredients that could represent the transformed cusine
        ing = random.choice(chn_ing)

        action_verbs = self.get_action_verbs()
        i = 0
        # check if certain cooking methods exist in the steps. If there are,
        # then add the ingredients into that step
        flag = False
        for i in range(len(action_verbs)):
            for verb in action_verbs[i]:
                if verb.lower() in cooking_methods:
                    flag = True
            if flag:
                break
        if flag:
            step_ing = self.steps_copy[i]
            step_ing.ingredients = list(step_ing.ingredients)
            step_ing.ingredients.append("20g of " + ing)
            step_ing.action = "Add chopped 20g of " + ing + " and mix with other ingredients. " + step_ing.action
            transformed_recipe.ingredients.append(ing)

        for step in self.steps_copy:
            print("ingredients: ", step.ingredients)
            print("action: ", step.action)

        return transformed_recipe
        
if __name__ == "__main__":
	pass