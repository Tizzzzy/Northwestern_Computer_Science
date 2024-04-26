from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction

from structure import Step, Recipe
from utils import extract_action_noun_phrases, parse_number_or_ordinal, parse_mixed_ordinal

HOW_PRE_PHRASES = ["i do not know how to ", "i do not know how ", "i don't know how to ", "i don't know how ", 
                  "i dont know how to ", "i dont know how ", "idk how to ", "idk how ", "how do i ", "how to ", "how "]
HOW_POST_PHRASES = [" works"," work"]
HOW_SUB_PHRASES = ["does that", "does this", "does it", "do that", "do this", "do it", "that", "this", "it"]

WHAT_PRE_PHRASES = ["i do not know what ", "i don't know what ", "idk what ", "what "]

COOK_VERBS = ['simmer', 'whip', 'preheat', 'shred', 'stir-fry', 'whisk', 'poach', 'fry', 'pare', 'fold in', 'dice', 'cream', 
              'scald', 'baste', 'stew', 'chop', 'brown', 'saute', 'dilute', 'heat', 'bake', 'marinate', 'mince', 'mix', 'deep fry', 
              'grill', 'blend', 'roast', 'grease', 'slice', 'puree', 'score', 'stir', 'beat', 'drain', 'braise', 'season', 'sauté', 
              'combine', 'steam', 'cook', 'sift', 'gather', 'knead', 'cube', 'blanch', 'stir fry', 'garnish', 'boil']


QUERY_URL = "https://www.google.com/search?q="

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


class ActionProcessRecipe(Action):
    def name(self) -> str:
        return "action_process_recipe"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        # Extracting the 'url' entity
        url = next(tracker.get_latest_entity_values("url"), None)

        if url is not None:
            dispatcher.utter_message(text=f"Got it! The URL is {url}. What do you want to do next?\n\t(1) Show all ingredients\n\t(2) Show basic information\n\t(3) Go to Step 1")
            global recipe
            recipe = Recipe(url)
            recipe.load()
            return [SlotSet("cur_step", 1)]
            # return [SlotSet("recipe_url", url), SlotSet("ingredients", recipe.ingredients),
            #         SlotSet("actions", recipe.actions), SlotSet("utensils", recipe.utensils),
            #         SlotSet("tools", recipe.tools), SlotSet("other_parameters", recipe.other_parameters)]
        else:
            # Handle the case where URL is not provided
            dispatcher.utter_message(text="Please provide a valid URL.")
            return []

class ActionOptionChoose(Action):
    def name(self) -> str:
        return "action_option_choose"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:
        option = next(tracker.get_latest_entity_values("option"), None)

        recipe_url = tracker.get_slot("recipe_url")

        if recipe_url:
            if option == "1":
                # ingredients = tracker.get_slot("ingredients")
                ingredients = recipe.ingredients
                msg = ""
                for i, ingredient in enumerate(ingredients):
                    msg += "Ingredient {}: {}\n".format(i + 1, ingredient)
                dispatcher.utter_message(text=msg)
                return []
                # return [FollowupAction("action1")]
            elif option == "2":
                # information = tracker.get_slot("other_parameters")
                information = recipe.other_parameters
                msg = ""
                for para in information:
                    for key in para:
                        msg += "{}: {}\n".format(key, para[key])
                dispatcher.utter_message(text=msg)
                return []
                # return [FollowupAction("action2")]
            elif option == "3":
                return [FollowupAction("action_current_step")]
                # return [FollowupAction("action2")]
            else:
                dispatcher.utter_message(text="Please choose a valid option.")
                return []
        else:
            dispatcher.utter_message(text="Please provide a valid URL first.")
            return []

class ActionAnswerHowToQuery(Action):

    def name(self) -> Text:
        return "action_answer_how_to_query"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        msg = ""
        question = " ".join(tracker.latest_message["text"].lower().split())
        if question[-1] == "?":
            question = question[:-1]
        is_parsed = False
        subs = None

        for p in HOW_PRE_PHRASES:
            if question[:len(p)] == p:
                question = question[len(p):]
                is_parsed = True
                break
        for p in HOW_POST_PHRASES:
            if question[-len(p):] == p:
                question = question[:-len(p)]
                is_parsed = True
                break
        for p in HOW_SUB_PHRASES:
            i = question.find(p)
            if i > -1:
                cur_step = tracker.get_slot("cur_step")
                if cur_step <= 0:
                    dispatcher.utter_message(text="Input a recipe URL, or go to step 1 first.")
                    return []
                step = recipe.steps[cur_step-1]
                subs = step.action.split(".")[:-1]
                if len(subs) == 1:
                    phrases = extract_action_noun_phrases(subs[0], COOK_VERBS)
                    print(phrases)
                    question = question[:i] + phrases[0].lower() + question[i + len(p):]
                    subs = None
                else:
                    sub_questions = []
                    for sub in subs:
                        phrases = extract_action_noun_phrases(sub, COOK_VERBS)
                        for phrase in phrases:
                            sub_questions.append(question[:i] + phrase.lower() + question[i + len(p):])
                break
        
        if subs is None:
            q = "+".join(question.split())
            msg = "No worries. I found a reference for you: {}{}\n".format(QUERY_URL + ("how+to+" if is_parsed else ""), q)
        elif len(subs) == 0:
            msg = "I didn't undestand your question. Try asking about a specific ingredient/tool/cooking action you're curious about."
        else:
            msg = "Could you be more specific? Or you can refer to the following possible references:\n"
            for qq in sub_questions:
                q = "+".join(qq.split())
                msg += "\t{}{}\n".format(QUERY_URL + ("how+to+" if is_parsed else ""), q)
        dispatcher.utter_message(text=msg)
        return []

class ActionAnswerWhatQuery(Action):

    def name(self) -> Text:
        return "action_answer_what_query"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        msg = ""
        question = " ".join(tracker.latest_message["text"].lower().split())
        if question[-1] == "?":
            question = question[:-1]

        for p in WHAT_PRE_PHRASES:
            if question[:len(p)] == p:
                question = question[len(p):]
                break
        q = "+".join(question.split())
        msg = "No worries. I found a reference for you: {}{}\n".format(QUERY_URL + "what+", q)
        dispatcher.utter_message(text=msg)
        return []

class ActionNextStep(Action):

    def name(self) -> Text:
        return "action_next_step"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")
        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL, or go to step 1 first.")
            return []
        elif cur_step == len(recipe.steps):
            dispatcher.utter_message(text="You have reached the last step.")
            return []
        else:
            step = recipe.steps[cur_step]
            msg = "Step {}:\n".format(cur_step+1)
            msg += step.action + "\n"
            msg += "Ingredients: {}\n".format(", ".join(step.ingredients))
            msg += "Tools: {}\n".format(", ".join(step.tools))
            msg += "Time: {} minutes\n".format(step.time)
            dispatcher.utter_message(text=msg)
            return [SlotSet("cur_step", cur_step+1)]

class ActionPrevStep(Action):

    def name(self) -> Text:
        return "action_prev_step"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")
        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        elif cur_step == 1:
            dispatcher.utter_message(text="You have reached the first step.")
            return []
        else:
            step = recipe.steps[cur_step-2]
            msg = "Step {}:\n".format(cur_step-1)
            msg += step.action + "\n"
            msg += "Ingredients: {}\n".format(", ".join(step.ingredients))
            msg += "Tools: {}\n".format(", ".join(step.tools))
            msg += "Time: {} minutes\n".format(step.time)
            dispatcher.utter_message(text=msg)
            return [SlotSet("cur_step", cur_step-1)]

class ActionCurrentStep(Action):

    def name(self) -> Text:
        return "action_current_step"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        else:
            step = recipe.steps[cur_step-1]
            msg = "Step {}:\n".format(cur_step)
            msg += step.action + "\n"
            msg += "Ingredients: {}\n".format("; ".join(step.ingredients))
            msg += "Tools: {}\n".format(", ".join(step.tools))
            msg += "Time: {} minutes\n".format(step.time)
            dispatcher.utter_message(text=msg)
            return []

class ActionDisplayIngredients(Action):

    def name(self) -> Text:
        return "action_display_ingredients"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        else:
            step = recipe.steps[cur_step-1]
            msg = "Ingredients: {}\n".format("; ".join(step.ingredients))
            dispatcher.utter_message(text=msg)
            return []

class ActionDisplayAllIngredients(Action):

    def name(self) -> Text:
        return "action_display_all_ingredients"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        else:
            ingredients = recipe.ingredients
            msg = ""
            for i, ingredient in enumerate(ingredients):
                msg += "Ingredient {}: {}\n".format(i + 1, ingredient)
            dispatcher.utter_message(text=msg)
            return []

class ActionDisplayTools(Action):

    def name(self) -> Text:
        return "action_display_tools"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        else:
            step = recipe.steps[cur_step-1]
            msg = "Tools: {}\n".format(", ".join(step.tools))
            dispatcher.utter_message(text=msg)
            return []

class ActionDisplayAllTools(Action):

    def name(self) -> Text:
        return "action_display_all_tools"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        else:
            tools = recipe.tools
            msg = ""
            for i, tool in enumerate(tools):
                msg += "Tool {}: {}\n".format(i + 1, tool)
            dispatcher.utter_message(text=msg)
            return []

class ActionDisplayTime(Action):

    def name(self) -> Text:
        return "action_display_time"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        else:
            step = recipe.steps[cur_step-1]
            msg = "Time: {} minutes\n".format(step.time)
            dispatcher.utter_message(text=msg)
            return []

class ActionDisplayActions(Action):

    def name(self) -> Text:
        return "action_display_actions"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        else:
            step = recipe.steps[cur_step-1]
            msg = "Methods: {}\n".format(step.methods)
            msg += "Action: {}\n".format(step.action)
            dispatcher.utter_message(text=msg)
            return []

class ActionRefresh(Action):

    def name(self) -> Text:
        return "action_refresh"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        recipe = None

        return [SlotSet("cur_step", -1), SlotSet("recipe_url", None), FollowupAction("utter_goodbye")]

class ActionTransformVegan(Action):

    def name(self) -> Text:
        return "action_transform_vegan"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []

        global recipe
        new_recipe = recipe.transform_to_vegan(recipe)
        recipe = new_recipe

        dispatcher.utter_message(text="OK, it's been transformed to vegan.")

        return [SlotSet("cur_step", 1)]


class ActionTransformUnhealthy(Action):

    def name(self) -> Text:
        return "action_transform_unhealthy"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []

        global recipe
        new_recipe = recipe.transform_to_unhealthy(recipe)
        recipe = new_recipe
        
        dispatcher.utter_message(text="OK, it's been transformed to unhealthy.")

        return [SlotSet("cur_step", 1)]

class ActionTransformSoutheastAsian(Action):

    def name(self) -> Text:
        return "action_transform_southeast_asian"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []

        global recipe
        new_recipe = recipe.transform_to_southeast_asian(recipe)
        recipe = new_recipe

        dispatcher.utter_message(text="OK, it's been transformed to Southeast Asian style.")

        return [SlotSet("cur_step", 1)]

class ActionTransformNonVegan(Action):

    def name(self) -> Text:
        return "action_transform_non_vegan"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:

        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []

        global recipe
        new_recipe = recipe.transform_from_vegan(recipe)
        recipe = new_recipe
        dispatcher.utter_message(text="OK, it's been transformed to non-vegan.")

        return [SlotSet("cur_step", 1)]

class ActionTransformChinese(Action):

    def name(self) -> Text:
        return "action_transform_chinese"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:

        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []

        global recipe
        new_recipe = recipe.transform_to_Chinese(recipe)
        recipe = new_recipe
        dispatcher.utter_message(text="OK, it's been transformed to Chinese.")

        return [SlotSet("cur_step", 1)]

class ActionTransformBackToOriginal(Action):

    def name(self) -> Text:
        return "action_transform_back_to_original"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:
        
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []

        global recipe
        if recipe.org_recipe:
            new_recipe = recipe.org_recipe
            recipe = new_recipe
            return [SlotSet("cur_step", 1)]
        else:
            dispatcher.utter_message(text="This recipe is already the original version.")
            return []

class ActionNavigateSteps(Action):
    def name(self):
        return "action_navigate_steps"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text,Any]]:

        step_entity = next(tracker.get_latest_entity_values("step_number"), None)
        cur_step = tracker.get_slot("cur_step")

        if cur_step < 0:
            dispatcher.utter_message(text="Input a recipe URL first.")
            return []
        
        if step_entity is not None:
            step_number = parse_number_or_ordinal(step_entity)
            if isinstance(step_number, int):
                if step_number <= 0 or step_number > len(recipe.steps):
                    dispatcher.utter_message(text="Please choose a valid step number in [1, {}]".format(len(recipe.steps)))
                else:
                    dispatcher.utter_message(text=f"Taking you to step {step_number}.")
                    return [SlotSet("cur_step", step_number), FollowupAction("action_current_step")]
            else:
                dispatcher.utter_message(text="I'm not sure which step you want to go to.")
        else:
            dispatcher.utter_message(text="I'm not sure which step you want to go to. 22213")

        return []

