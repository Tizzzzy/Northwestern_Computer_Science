# HW3—Knowledge Base: Part 2


## Introduction

In this part of the homework, you are going to extend the knowledge base (KB) and create an inference engine. The KB will now support three main interfaces:

- `Assert`: Adds facts and rules to the KB. After you add facts and rules to the KB, the Forward Chaining algorithm is used to infer other facts and rules.
- `Ask`: Asks queries and returns a list of bindings for facts and rules.
- `Retract`: Removes asserted facts and rules from the KB. Also, removes all other facts and rules that are dependent on the removed fact and rule.

The end result of this assignment is a KB that can be used to model a world/game/thing with a static set of rules. Most board games and established businesses (during a short period of time) fell into this category. Asserted facts and rules can be treated as factual observations about the state of the world/game/thing, situations which hold until they cease to be valid, when they are retracted by us, the users. Inferred facts and rules should be thought of as hypotheses: things you write on a scratch paper when solving a problem. They should be generated when there are enough supporting evidences and removed as soon as *any* supporting evidence is retracted.

## Starter code

We provide you five files with code, details of which are described at the end of this writeup:

- `main.py`: Contains code for testing the KB, which will be implemented as the `KnowledgeBase` class
- `student_code.py`: Contains the `KnowledgeBase` and `InferenceEngine` classes and is where you will be writing code.
- `logical_classes.py`: Contains classes for each type of logical component, e.g., `Fact`, `Rule`, etc.
- `util.py`: Contains several useful helper functions
- `read.py`: Contains functions that read statements from files or terminal. (You won't need to read/explore this file.))

There are also two data files that contain the facts and rules to be inserted into the KB:

- `statements_kb.txt`
- `statements_kb2.txt`

Some of the provided tests use `statements_kb.txt` while others add facts and rules to the KB from within the tests by first reading them in using read.parse_input(). And you may use `statements_kb2.txt` to generate your own tests.

## Your task

To get you started, the `Assert` and `Ask` interfaces have been written - exposed via the `KnowledgeBase.kb_assert` and `KnowledgeBase.kb_ask` methods.

Your task is two-part:

   1. Implement the Forward Chaining inferences that occur upon asserting facts and rules into the KB - i.e., implement the `InferenceEnginer.fc_infer` method.
   2. Implement the `Retract` interface to remove facts and rules from the KB - i.e., implement the `KnowledgeBase.kb_retract` method.

### Rule currying in `fc_infer`

The key idea is that we don't just infer new facts - we can also infer new rules.

When we add a new fact to the KB, we check to see if it triggers any rule(s). When we add a new rule, we check to see if it's triggered by existing facts.

However, a rule might have multiple statements on its left-hand side (LHS), and we don't want to iterate each of these statements every time we add a new fact to the KB. Instead, we'll employ a cool trick. Whenever we add a new rule, we'll only check the first element of the LHS of that rule against the facts in our KB. (If we add a new fact, we'll reverse this - we'll examine each rule in our KB and check the first element of its LHS against this new fact.) If there's a match with this first element, we'll add a new rule paired with *bindings* for that match.

For example, imagine a box-world. Consider a rule stating that if a box `?x` is larger than another box `?y`, and box `?x` is on box `?y`, then box `?y` is covered. Formally, that looks like:

```
((sizeIsLess(?y, ?x), on(?x, ?y)) => covered(?y))
```

Now imagine that we know that box `A` is bigger than box `B`; i.e., we have the fact `sizeIsLess(B, A)` in the KB. The above rule then matches, with the bindings `((?x: A, ?y: B))`. With that binding in place, we can now infer a new rule that uses it:

```
(on(A, B)) => covered(B)
```

If we find the fact `on(A, B)` in the KB, then we can use this rule to infer the fact `covered(B)`. If we don't have that fact, however, we now have a simple rule that will let us make the inference easily if we see that fact in the future.

### Removing facts and rules inferred from a removed fact or rule

When you remove a fact or a rule, you also need to remove all facts and rules that were inferred using this fact or rule. However, a given fact or rule might be supported by multiple facts and rules - so, you'll need to check whether the facts and rules rules inferred from this fact or rule are also supported by other facts or rules (or if they were directly asserted).

As a simplification, you can assume that **no rules will create circular dependencies**. E.g., imagine a situation like `A => B`, `B => C`, and `C => B`. Removing `A` would mean removing `B` and `C`, since they depend on `A` via those rules. However, implementing that would get messy, since `B` and `C` depend on each other. You will **NOT** be given scenarios like this.

### Testing

To grade this homework, we'll run several test cases similar to the ones provided. In each test case, facts and rules will be asserted one by one into the KB, and additional operations will be performed on the populated KB. Some test cases focus more on testing `fc_infer` while others `kb_retract`. Note that the test cases provided to you with this part of the homework are significantly less comprehensive than those from Part 1. **It is, therefore, imperative that you make your own testing files and test cases.** Please feel free to share them on Campuswire. When sharing tests, please provide your rationale for each test, explain what you hope to test with it, and/or describe how you developed the test.

### Hints

#### Implementing `fc_infer`

- Use the `util.match` function to do unification and create possible bindings.
- Use the `util.instantiate` function to bind a variable in the rest of a rule.
- `Rule`s and `Fact`s have fields for `supported_by`, `supports_facts`, and `supports_rules`. Use them to track inferences! For example, imagine that a fact `F` and a rule `R` matched to infer a new fact/rule `fr`.
  - `fr` is *supported* by `F` and `R`. Add them to `fr`'s `supported_by` list of lists - you can do this by passing them as a constructor argument when creating `fr`.
  - `F` and `R` now *support* `fr`. Add `fr` to the `supports_rules` and `supports_facts` lists (as appropriate) in `F` and `R`.

#### Implementing `kb_retract`

- Asserted facts and rules that DO NOT have support can be retracted.

- Asserted facts and rules that have support cannot be retracted; when attempted to be retracted, they can and should only be unasserted.

- Inferred facts and rules that DO NOT have support shouldn’t even be existing and they should have been retracted when their supports were retracted.

- Inferred facts and rules that have support cannot be retracted, and there’s no unasserting needed, as they are already unasserted. In other words, when inferred facts and rules are attempted to be retracted, nothing should be done to them. They _can_, however, be asserted after they have already been inferred, which then will give them stronger protection against retraction (as they would have to become both unasserted and unsupported to be retracted). This is not required by this homework, though, and is just something to know.

- Use the `supports_rules` and `supports_facts` fields to find and adjust facts and rules that are supported by a retracted fact or rule. For every fact or rule that the retracted fact or rule supports:
  - Its corresponding list in the `supported_by` list of lists needs to be adjusted accordingly.
  - If it is no longer supported as a result of the retraction and if it is also not asserted, it should be removed.

## Appendix: File Breakdown

Below is a description of each included file and the classes contained within each including a listing of their attributes. Each file has documentation in the code reflecting the information below (in most cases they are exactly the same). As you read through the attributes, follow along in the corresponding files, and make sure you're understanding the descriptions.

Attributes of each class are listed in the following format (__Note:__ If you see a type like `Fact|Rule` the `|` type is `or` and means that the type can be either `Fact` or `Rule`):

- `field_name` (`type`) - text description

### `logical_classes.py`

This file defines all basic structure classes.

### Fact

Represents a fact in our knowledge base (KB). Has a statement containing the content of the fact, e.g., `(isa Sorceress Wizard)`, and fields tracking which facts and rules in the KB it supports and is supported by.

**Attributes**

- `name` (`str`): 'fact', the name of this class
- `statement` (`Statement`): statement of this fact, basically what the fact actually says
- `asserted` (`bool`): flag indicating if fact was asserted instead of inferred from other facts and rules in the KB
- `supported_by` (`listof listof Fact|Rule`): Facts/Rules that allow inference of the statement
- `supports_facts` (`listof Fact`): Facts that this fact supports
- `supports_rules` (`listof Rule`): Rules that this fact supports

### Rule

Represents a rule in our KB. Has a list of statements (the left-hand side or LHS) containing the statements that need to be in our KB for us to infer the right-hand-side or RHS statement. Also has fields tracking which facts and rules in the KB it supports and is supported by.

**Attributes**

- `name` (`str`): 'rule', the name of this class
- `lhs` (`listof Statement`): LHS statements of this rule
- `rhs` (`Statement`): RHS statement of this rule
- `asserted` (`bool`): flag indicating if rule was asserted instead of inferred from other facts and rules in the KB
- `supported_by` (`listof listof Fact|Rule`): Facts/Rules that allow inference of the statement
- `supports_facts` (`listof Fact`): Facts that this rule supports
- `supports_rules` (`listof Rule`): Rules that this rule supports

### Statement

Represents a statement in our KB, e.g., `(attacked Ai Nosliw)`, `(diamonds Loot)`, `(isa Sorceress Wizard)`, etc. These statements show up in Facts or on the LHS and RHS of Rules.

**Attributes**

- `predicate` (`str`) - the predicate of the statement, e.g., `isa`, `hero`, `needs`
- `terms` (`listof Term`) - list of terms (Variable or Constant) in the statement, e.g., `'Nosliw'` or `'?d'`

### Term

Represents a term (a Variable or a Constant) in our KB. It could be thought of as a super class of Variable and Constant, though there is no actual inheritance implemented in the code.

**Attributes**

- `term` (`Variable|Constant`) - the Variable or Constant that this term holds (represents)

### Variable

Represents a variable used in statements, e.g., `?x`.

**Attributes**

- `element` (`str`): the name of the variable, e.g., `'?x'`

### Constant

Represents a constant used in statements.

**Attributes**

- `element` (`str`): the value of the constant, e.g., `'Nosliw'`

### Binding

Represents a binding of a constant to a variable, e.g., `'Nosliw'` might be bound to `'?d'`.

**Attributes**

- `variable` (`str`): the name of the variable associated with this binding, e.g., `'?d'`
- `constant` (`str`): the value of the variable, e.g., `'Nosliw'`

### Bindings

Represents Binding(s) used while matching two statements.

**Attributes**

- `bindings` (`listof Bindings`) - bindings involved in match
- `bindings_dict` (`dictof Bindings`) - bindings involved in match where key is bound variable and value is bound value, e.g., `some_bindings.bindings_dict['?d'] => 'Nosliw'`

**Methods**

- `add_binding(variable, value)` (`(Variable, Constant) => void`) - Add a binding from a variable to a value.
- `bound_to(variable)` (`(Variable) => Variable|Constant|False`) - Check if variable is bound. If so, return value bound to it, else False.
- `test_and_bind(variable_verm,value_term)` (`(Term, Term) => bool`) - Check if variable_term already bound. If so, return whether or not passed-in value_term matches bound value. If not, add binding between variable_terma and value_term, and return True.

### ListOfBindings

Container for multiple Bindings

**Methods**

- `add_bindings(bindings, facts_rules)` - (`(Bindings, listof Fact|Rule) => void`) - Add given bindings to list of Bindings along with associated facts or rules.

## `read.py`

This file has no classes but defines useful helper functions for reading input from the user or a file.

**Functions**

- `read_tokenize(file)` - (`(str) => (listof Fact, listof Rule)`) - Takes a filename, reads the file, and returns a fact list and a rule list.
- `parse_input(e)` - (`(str) => (int, str | listof str)`) - Parses input (cleaning it as it does so), assigning labels and splitting rules into LHS and RHS.
- `read_from_input(message)` - (`(str) => str`) - Collects user input from the command line.
- `get_new_fact_or_rule()` - (`() => Fact | Rule`) - Gets a new fact or rule by typing, nothing passed in, data comes from user input.
- `get_new_statements()` - (`() => listof Statement`) - Reads statements from input, nothing passed in, data comes from user input.

## `util.py`

This file has no classes but defines useful helper functions.

**Functions**

- `is_var(var)` (`(str|Variable|Constant|Term) => bool`) - Check whether an element is a variable (either instance of Variable, instance of Term (where .term is a Variable) or a string starting with `'?'`, e.g., `'?d'`).
- `match(state1, state2, bindings=None)` (`(Statement, Statement, Bindings) => Bindings|False`) - Match two statements, and return the associated bindings or False if there is no binding.
- `match_recursive(terms1, terms2, bindings)` (`(listof Term, listof Term, Bindings) => Bindings|False`) - recursive helper for match
- `instantiate(statement, bindings)` (`(Statement, Bindings) => Statement|Term`)  - Generate Statement from given statement and bindings. Constructed statement has bound values for variables if they exist in bindings.
- `printv(message, level, verbose, data=[])` (`(str, int, int, listof any) => void`) - Prints message if verbose > level. If data provided, then formats message with given data.

## `student_code.py`

This file defines the two classes you must implement, `KnowledgeBase` and `InferenceEngine`.

### KnowledgeBase

Represents a knowledge base and contains the two methods described in the writeup (`Assert` and `Ask`).

### InferenceEngine

Represents an inference engine. Implements Forward Chaining in this homework.
