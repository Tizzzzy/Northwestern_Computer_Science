import unittest
import student_code as sc
import expand
import sys, signal

time_map1 = {
    'John_Stevens':{ 'John_Stevens':None, 'John_Doe':1, 'Kim_Lee':1, 'Raj_Gupta':None, 'Walter_Walker':1, 'Alex_Robbinson':None, 'Mariana_Cardoso':None },
    'John_Doe':{ 'John_Stevens':1, 'John_Doe':None, 'Kim_Lee':1, 'Raj_Gupta':1, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':None },
    'Kim_Lee':{ 'John_Stevens':1, 'John_Doe':1, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':None },
    'Raj_Gupta':{ 'John_Stevens':None, 'John_Doe':1, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':1 },
    'Walter_Walker':{ 'John_Stevens':1, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':1, 'Mariana_Cardoso':None },
    'Alex_Robbinson':{ 'John_Stevens':None, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':1, 'Alex_Robbinson':None, 'Mariana_Cardoso':1 },
    'Mariana_Cardoso':{ 'John_Stevens':None, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':1, 'Walter_Walker':None, 'Alex_Robbinson':1, 'Mariana_Cardoso':None }
}

dis_map2 = {
    'John_Stevens':{ 'John_Stevens':0, 'John_Doe':3, 'Kim_Lee':5, 'Raj_Gupta':5, 'Walter_Walker':1, 'Alex_Robbinson':2, 'Mariana_Cardoso':12 },
    'John_Doe':{ 'John_Stevens':3, 'John_Doe':0, 'Kim_Lee':3, 'Raj_Gupta':3, 'Walter_Walker':4, 'Alex_Robbinson':5, 'Mariana_Cardoso':8 },
    'Kim_Lee':{ 'John_Stevens':5, 'John_Doe':3, 'Kim_Lee':0, 'Raj_Gupta':8, 'Walter_Walker':5, 'Alex_Robbinson':7, 'Mariana_Cardoso':12 },
    'Raj_Gupta':{ 'John_Stevens':5, 'John_Doe':3, 'Kim_Lee':8, 'Raj_Gupta':0, 'Walter_Walker':7, 'Alex_Robbinson':7, 'Mariana_Cardoso':2 },
    'Walter_Walker':{ 'John_Stevens':1, 'John_Doe':4, 'Kim_Lee':5, 'Raj_Gupta':7, 'Walter_Walker':0, 'Alex_Robbinson':1, 'Mariana_Cardoso':15 },
    'Alex_Robbinson':{ 'John_Stevens':2, 'John_Doe':5, 'Kim_Lee':7, 'Raj_Gupta':7, 'Walter_Walker':1, 'Alex_Robbinson':0, 'Mariana_Cardoso':12 },
    'Mariana_Cardoso':{ 'John_Stevens':12, 'John_Doe':8, 'Kim_Lee':12, 'Raj_Gupta':2, 'Walter_Walker':15, 'Alex_Robbinson':12, 'Mariana_Cardoso':0 } }

time_map2 = {
    'John_Stevens':{ 'John_Stevens':None, 'John_Doe':28, 'Kim_Lee':13, 'Raj_Gupta':None, 'Walter_Walker':11, 'Alex_Robbinson':None, 'Mariana_Cardoso':None },
    'John_Doe':{ 'John_Stevens':14, 'John_Doe':None, 'Kim_Lee':14, 'Raj_Gupta':13, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':None },
    'Kim_Lee':{ 'John_Stevens':14, 'John_Doe':14, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':None },
    'Raj_Gupta':{ 'John_Stevens':None, 'John_Doe':14, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':12 },
    'Walter_Walker':{ 'John_Stevens':11, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':11, 'Mariana_Cardoso':None },
    'Alex_Robbinson':{ 'John_Stevens':None, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':12, 'Alex_Robbinson':None, 'Mariana_Cardoso':15 },
    'Mariana_Cardoso':{ 'John_Stevens':None, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':13, 'Walter_Walker':None, 'Alex_Robbinson':15, 'Mariana_Cardoso':None } }


dis_map5 = {
    'John_Stevens':{ 'John_Stevens':0, 'John_Doe':3, 'Kim_Lee':5, 'Raj_Gupta':5, 'Walter_Walker':1, 'Alex_Robbinson':2, 'Mariana_Cardoso':12, 'Sarah_Parker': 4 },
    'John_Doe':{ 'John_Stevens':3, 'John_Doe':0, 'Kim_Lee':3, 'Raj_Gupta':3, 'Walter_Walker':4, 'Alex_Robbinson':5, 'Mariana_Cardoso':8, 'Sarah_Parker': 3 },
    'Kim_Lee':{ 'John_Stevens':5, 'John_Doe':3, 'Kim_Lee':0, 'Raj_Gupta':8, 'Walter_Walker':5, 'Alex_Robbinson':7, 'Mariana_Cardoso':12, 'Sarah_Parker': 6 },
    'Raj_Gupta':{ 'John_Stevens':5, 'John_Doe':3, 'Kim_Lee':8, 'Raj_Gupta':0, 'Walter_Walker':7, 'Alex_Robbinson':7, 'Mariana_Cardoso':2, 'Sarah_Parker': 4 },
    'Walter_Walker':{ 'John_Stevens':1, 'John_Doe':4, 'Kim_Lee':5, 'Raj_Gupta':7, 'Walter_Walker':0, 'Alex_Robbinson':1, 'Mariana_Cardoso':15, 'Sarah_Parker': 5 },
    'Alex_Robbinson':{ 'John_Stevens':2, 'John_Doe':5, 'Kim_Lee':7, 'Raj_Gupta':7, 'Walter_Walker':1, 'Alex_Robbinson':0, 'Mariana_Cardoso':12 , 'Sarah_Parker': 4},
    'Mariana_Cardoso':{ 'John_Stevens':12, 'John_Doe':8, 'Kim_Lee':12, 'Raj_Gupta':2, 'Walter_Walker':15, 'Alex_Robbinson':12, 'Mariana_Cardoso':0, 'Sarah_Parker': 5 },
    'Sarah_Parker': { 'John_Stevens':12, 'John_Doe':8, 'Kim_Lee':12, 'Raj_Gupta':2, 'Walter_Walker':15, 'Alex_Robbinson':12, 'Mariana_Cardoso':0, 'Sarah_Parker': 0}}

time_map5 = {
    'John_Stevens':{ 'John_Stevens':None, 'John_Doe':28, 'Kim_Lee':13, 'Raj_Gupta':None, 'Walter_Walker':11, 'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Sarah_Parker': None },
    'John_Doe':{ 'John_Stevens':14, 'John_Doe':None, 'Kim_Lee':14, 'Raj_Gupta':13, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Sarah_Parker': 6 },
    'Kim_Lee':{ 'John_Stevens':14, 'John_Doe':14, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Sarah_Parker': None },
    'Raj_Gupta':{ 'John_Stevens':None, 'John_Doe':14, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':12, 'Sarah_Parker': None },
    'Walter_Walker':{ 'John_Stevens':11, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':None, 'Alex_Robbinson':11, 'Mariana_Cardoso':None, 'Sarah_Parker': None },
    'Alex_Robbinson':{ 'John_Stevens':None, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':None, 'Walter_Walker':12, 'Alex_Robbinson':None, 'Mariana_Cardoso':15, 'Sarah_Parker': None },
    'Mariana_Cardoso':{ 'John_Stevens':None, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':13, 'Walter_Walker':None, 'Alex_Robbinson':15, 'Mariana_Cardoso':None, 'Sarah_Parker': None },
    'Sarah_Parker': { 'John_Stevens':7, 'John_Doe':None, 'Kim_Lee':None, 'Raj_Gupta':6, 'Walter_Walker':None, 'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Sarah_Parker': None } }

dis_mapM = {
    'Alex_Robbinson': { 'Alex_Robbinson':0, 'Benjamin_Walker':1, 'Catherine_Stevens':2, 'David_Stone':3, 'Elena_Cardoso':1, 'Fiona_Rutherfurd':2, 'George_Richford':3, 'Hannah_Mullard':4, 'Ivy_Doe':2, 'Jessica_Baker':3, 'Kim_Lee':4, 'Lakshmi_Raman':5, 'Mariana_Cardoso':3, 'Heather_Raymond':4, 'Olivia_Keeton':5, 'Peter_Kilshaw':6},
    'Benjamin_Walker': { 'Alex_Robbinson':1, 'Benjamin_Walker':0, 'Catherine_Stevens':1, 'David_Stone':2, 'Elena_Cardoso':2, 'Fiona_Rutherfurd':1, 'George_Richford':2, 'Hannah_Mullard':3, 'Ivy_Doe':3, 'Jessica_Baker':2, 'Kim_Lee':3, 'Lakshmi_Raman':4, 'Mariana_Cardoso':4, 'Heather_Raymond':3, 'Olivia_Keeton':4, 'Peter_Kilshaw':5},
    'Catherine_Stevens': { 'Alex_Robbinson':2, 'Benjamin_Walker':1, 'Catherine_Stevens':0, 'David_Stone':1, 'Elena_Cardoso':3, 'Fiona_Rutherfurd':2, 'George_Richford':1, 'Hannah_Mullard':2, 'Ivy_Doe':4, 'Jessica_Baker':3, 'Kim_Lee':2, 'Lakshmi_Raman':3, 'Mariana_Cardoso':5, 'Heather_Raymond':4, 'Olivia_Keeton':3, 'Peter_Kilshaw':4},
    'David_Stone': { 'Alex_Robbinson':3, 'Benjamin_Walker':2, 'Catherine_Stevens':1, 'David_Stone':0, 'Elena_Cardoso':4, 'Fiona_Rutherfurd':3, 'George_Richford':2, 'Hannah_Mullard':1, 'Ivy_Doe':5, 'Jessica_Baker':4, 'Kim_Lee':3, 'Lakshmi_Raman':2, 'Mariana_Cardoso':6, 'Heather_Raymond':5, 'Olivia_Keeton':4, 'Peter_Kilshaw':3},
    'Elena_Cardoso': { 'Alex_Robbinson':1, 'Benjamin_Walker':2, 'Catherine_Stevens':3, 'David_Stone':4, 'Elena_Cardoso':0, 'Fiona_Rutherfurd':1, 'George_Richford':2, 'Hannah_Mullard':3, 'Ivy_Doe':1, 'Jessica_Baker':2, 'Kim_Lee':3, 'Lakshmi_Raman':4, 'Mariana_Cardoso':2, 'Heather_Raymond':3, 'Olivia_Keeton':4, 'Peter_Kilshaw':5},
    'Fiona_Rutherfurd': { 'Alex_Robbinson':2, 'Benjamin_Walker':1, 'Catherine_Stevens':2, 'David_Stone':3, 'Elena_Cardoso':1, 'Fiona_Rutherfurd':0, 'George_Richford':1, 'Hannah_Mullard':2, 'Ivy_Doe':2, 'Jessica_Baker':1, 'Kim_Lee':2, 'Lakshmi_Raman':3, 'Mariana_Cardoso':3, 'Heather_Raymond':2, 'Olivia_Keeton':3, 'Peter_Kilshaw':4},
    'George_Richford': { 'Alex_Robbinson':3, 'Benjamin_Walker':2, 'Catherine_Stevens':1, 'David_Stone':2, 'Elena_Cardoso':2, 'Fiona_Rutherfurd':1, 'George_Richford':0, 'Hannah_Mullard':1, 'Ivy_Doe':3, 'Jessica_Baker':2, 'Kim_Lee':1, 'Lakshmi_Raman':2, 'Mariana_Cardoso':4, 'Heather_Raymond':3, 'Olivia_Keeton':2, 'Peter_Kilshaw':3},
    'Hannah_Mullard': { 'Alex_Robbinson':4, 'Benjamin_Walker':3, 'Catherine_Stevens':2, 'David_Stone':1, 'Elena_Cardoso':3, 'Fiona_Rutherfurd':2, 'George_Richford':1, 'Hannah_Mullard':0, 'Ivy_Doe':4, 'Jessica_Baker':3, 'Kim_Lee':2, 'Lakshmi_Raman':1, 'Mariana_Cardoso':5, 'Heather_Raymond':4, 'Olivia_Keeton':2, 'Peter_Kilshaw':2},
    'Ivy_Doe': { 'Alex_Robbinson':2, 'Benjamin_Walker':3, 'Catherine_Stevens':4, 'David_Stone':5, 'Elena_Cardoso':1, 'Fiona_Rutherfurd':2, 'George_Richford':3, 'Hannah_Mullard':4, 'Ivy_Doe':0, 'Jessica_Baker':1, 'Kim_Lee':2, 'Lakshmi_Raman':3, 'Mariana_Cardoso':1, 'Heather_Raymond':2, 'Olivia_Keeton':3, 'Peter_Kilshaw':4},
    'Jessica_Baker': { 'Alex_Robbinson':3, 'Benjamin_Walker':2, 'Catherine_Stevens':3, 'David_Stone':4, 'Elena_Cardoso':2, 'Fiona_Rutherfurd':1, 'George_Richford':2, 'Hannah_Mullard':3, 'Ivy_Doe':2, 'Jessica_Baker':0, 'Kim_Lee':1, 'Lakshmi_Raman':2, 'Mariana_Cardoso':2, 'Heather_Raymond':1, 'Olivia_Keeton':2, 'Peter_Kilshaw':3},
    'Kim_Lee': { 'Alex_Robbinson':4, 'Benjamin_Walker':3, 'Catherine_Stevens':2, 'David_Stone':3, 'Elena_Cardoso':3, 'Fiona_Rutherfurd':2, 'George_Richford':1, 'Hannah_Mullard':2, 'Ivy_Doe':2, 'Jessica_Baker':1, 'Kim_Lee':0, 'Lakshmi_Raman':1, 'Mariana_Cardoso':3, 'Heather_Raymond':2, 'Olivia_Keeton':1, 'Peter_Kilshaw':2},
    'Lakshmi_Raman': { 'Alex_Robbinson':5, 'Benjamin_Walker':4, 'Catherine_Stevens':3, 'David_Stone':2, 'Elena_Cardoso':4, 'Fiona_Rutherfurd':3, 'George_Richford':2, 'Hannah_Mullard':1, 'Ivy_Doe':3, 'Jessica_Baker':2, 'Kim_Lee':1, 'Lakshmi_Raman':0, 'Mariana_Cardoso':4, 'Heather_Raymond':3, 'Olivia_Keeton':2, 'Peter_Kilshaw':1},
    'Mariana_Cardoso': { 'Alex_Robbinson':3, 'Benjamin_Walker':4, 'Catherine_Stevens':5, 'David_Stone':6, 'Elena_Cardoso':2, 'Fiona_Rutherfurd':3, 'George_Richford':4, 'Hannah_Mullard':5, 'Ivy_Doe':1, 'Jessica_Baker':2, 'Kim_Lee':3, 'Lakshmi_Raman':4, 'Mariana_Cardoso':0, 'Heather_Raymond':1, 'Olivia_Keeton':2, 'Peter_Kilshaw':3},
    'Heather_Raymond': { 'Alex_Robbinson':4, 'Benjamin_Walker':3, 'Catherine_Stevens':4, 'David_Stone':5, 'Elena_Cardoso':3, 'Fiona_Rutherfurd':2, 'George_Richford':3, 'Hannah_Mullard':4, 'Ivy_Doe':2, 'Jessica_Baker':1, 'Kim_Lee':2, 'Lakshmi_Raman':3, 'Mariana_Cardoso':1, 'Heather_Raymond':0, 'Olivia_Keeton':1, 'Peter_Kilshaw':2},
    'Olivia_Keeton': { 'Alex_Robbinson':5, 'Benjamin_Walker':4, 'Catherine_Stevens':3, 'David_Stone':4, 'Elena_Cardoso':4, 'Fiona_Rutherfurd':3, 'George_Richford':2, 'Hannah_Mullard':3, 'Ivy_Doe':3, 'Jessica_Baker':2, 'Kim_Lee':1, 'Lakshmi_Raman':2, 'Mariana_Cardoso':2, 'Heather_Raymond':1, 'Olivia_Keeton':0, 'Peter_Kilshaw':1},
    'Peter_Kilshaw': { 'Alex_Robbinson':6, 'Benjamin_Walker':5, 'Catherine_Stevens':4, 'David_Stone':3, 'Elena_Cardoso':5, 'Fiona_Rutherfurd':4, 'George_Richford':3, 'Hannah_Mullard':2, 'Ivy_Doe':4, 'Jessica_Baker':3, 'Kim_Lee':2, 'Lakshmi_Raman':1, 'Mariana_Cardoso':3, 'Heather_Raymond':2, 'Olivia_Keeton':1, 'Peter_Kilshaw':0}
}

time_mapM = {
    'Alex_Robbinson': { 'Alex_Robbinson':None, 'Benjamin_Walker':1, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':1, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Benjamin_Walker': { 'Alex_Robbinson':1, 'Benjamin_Walker':None, 'Catherine_Stevens':1, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Catherine_Stevens': { 'Alex_Robbinson':None, 'Benjamin_Walker':1, 'Catherine_Stevens':None, 'David_Stone':1, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'David_Stone': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':1, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':1, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Elena_Cardoso': { 'Alex_Robbinson':1, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':1, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Fiona_Rutherfurd': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':1, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':1, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'George_Richford': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':1, 'George_Richford':None, 'Hannah_Mullard':1, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Hannah_Mullard': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':1, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':1, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Ivy_Doe': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':1, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':1, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Jessica_Baker': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':1, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':1, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Kim_Lee': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':1, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Lakshmi_Raman': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':1, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':1},
    'Mariana_Cardoso': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':1, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':None, 'Peter_Kilshaw':None},
    'Heather_Raymond': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':1, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':1, 'Peter_Kilshaw':None},
    'Olivia_Keeton': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':None, 'Mariana_Cardoso':None, 'Heather_Raymond':1, 'Olivia_Keeton':None, 'Peter_Kilshaw':1},
    'Peter_Kilshaw': { 'Alex_Robbinson':None, 'Benjamin_Walker':None, 'Catherine_Stevens':None, 'David_Stone':None, 'Elena_Cardoso':None, 'Fiona_Rutherfurd':None, 'George_Richford':None, 'Hannah_Mullard':None, 'Ivy_Doe':None, 'Jessica_Baker':None, 'Kim_Lee':None, 'Lakshmi_Raman':1, 'Mariana_Cardoso':None, 'Heather_Raymond':None, 'Olivia_Keeton':1, 'Peter_Kilshaw':None}
}

time_mapT = {
    'Alex_Robbinson': {'Alex_Robbinson':None, 'Mariana_Cardoso':1, 'Walter_Walker':1, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Mariana_Cardoso': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':1, 'Aaron_Stone':1, 'Catherine_Stevens':1, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Walter_Walker': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':1, 'Sarah_Parker':1, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Raj_Gupta': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Aaron_Stone': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Catherine_Stevens': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':1, 'Preeti_Singh':1, 'Kim_Lee':None},
    'Mike_Rhodes': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Sarah_Parker': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':1, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':1},
    'John_Stevens': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Preeti_Singh': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None},
    'Kim_Lee': {'Alex_Robbinson':None, 'Mariana_Cardoso':None, 'Walter_Walker':None, 'Raj_Gupta':None, 'Aaron_Stone':None, 'Catherine_Stevens':None, 'Mike_Rhodes':None, 'Sarah_Parker':None, 'John_Stevens':None, 'Preeti_Singh':None, 'Kim_Lee':None}
}



def interrupt(a,b):
    sys.exit(1)

class UnitTests(unittest.TestCase):

    def test1(self):
        expand.expand_count = 0
        path = sc.breadth_first_search(time_map1, 'John_Stevens', 'Mariana_Cardoso')
        # Two correct answers for right-to-left or left-to-right child traversal respectively
        self.assertIn(path, [['John_Stevens', 'Walter_Walker', 'Alex_Robbinson', 'Mariana_Cardoso'], ['John_Stevens', 'John_Doe', 'Raj_Gupta', 'Mariana_Cardoso']])
        self.assertIn(expand.expand_count, [6, 6])

    def test2(self):
         expand.expand_count = 0
         path = sc.breadth_first_search(time_mapM, 'Alex_Robbinson', 'George_Richford')
         self.assertEqual(path, ['Alex_Robbinson', 'Benjamin_Walker', 'Catherine_Stevens', 'David_Stone', 'Hannah_Mullard', 'George_Richford'])
         self.assertEqual(expand.expand_count, 8)

    def test3(self):
         expand.expand_count = 0
         path = sc.depth_first_search(time_mapT, 'Alex_Robbinson', 'Aaron_Stone')
         # Two correct answers for right-to-left or left-to-right child traversal respectively
         self.assertIn(path, [['Alex_Robbinson', 'Walter_Walker', 'Sarah_Parker', 'Aaron_Stone'], ['Alex_Robbinson', 'Mariana_Cardoso', 'Aaron_Stone']])
         self.assertIn(expand.expand_count, [4, 3])

    def test4(self):
         expand.expand_count = 0
         path = sc.depth_first_search(time_mapT, 'Alex_Robbinson', 'Raj_Gupta')
         self.assertEqual(path, ['Alex_Robbinson', 'Mariana_Cardoso', 'Raj_Gupta'])
         # Two correct answers for right-to-left or left-to-right child traversal respectively
         self.assertIn(expand.expand_count, [10, 2])

    def test5(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_map2, time_map2, 'John_Doe', 'Alex_Robbinson')
         self.assertEqual(path, ['John_Doe', 'John_Stevens', 'Walter_Walker', 'Alex_Robbinson'])
         self.assertEqual(expand.expand_count, 5)

    def test6(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_map2, time_map2, 'Mariana_Cardoso', 'John_Stevens')
         self.assertEqual(path, ['Mariana_Cardoso', 'Alex_Robbinson', 'Walter_Walker', 'John_Stevens'])
         self.assertEqual(expand.expand_count, 5)

    def test7(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_map5, time_map5, 'Sarah_Parker', 'John_Doe')
         self.assertEqual(path, ['Sarah_Parker', 'Raj_Gupta', 'John_Doe'])
         self.assertEqual(expand.expand_count, 3)

    def test8(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_map5, time_map5, 'Alex_Robbinson', 'Sarah_Parker')
         self.assertEqual(path, ['Alex_Robbinson', 'Mariana_Cardoso', 'Raj_Gupta', 'John_Doe', 'Sarah_Parker'])
         self.assertEqual(expand.expand_count, 7)

    def test9(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_map5, time_map5, 'John_Stevens', 'Sarah_Parker')
         self.assertEqual(path, ['John_Stevens', 'Kim_Lee', 'John_Doe', 'Sarah_Parker'])
         self.assertEqual(expand.expand_count, 5)

    def test10(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_mapM, time_mapM, 'Alex_Robbinson', 'Peter_Kilshaw')
         self.assertEqual(path, ['Alex_Robbinson', 'Benjamin_Walker', 'Catherine_Stevens', 'David_Stone', 'Hannah_Mullard', 'George_Richford', 'Fiona_Rutherfurd', 'Jessica_Baker', 'Heather_Raymond', 'Olivia_Keeton', 'Peter_Kilshaw'])
         self.assertEqual(expand.expand_count, 13)

    def test11(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_mapM, time_mapM, 'Hannah_Mullard', 'Peter_Kilshaw')
         self.assertEqual(path, ['Hannah_Mullard', 'George_Richford', 'Fiona_Rutherfurd', 'Jessica_Baker', 'Heather_Raymond', 'Olivia_Keeton', 'Peter_Kilshaw'])
         self.assertEqual(expand.expand_count, 8)

    def test12(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_mapM, time_mapM, 'Hannah_Mullard', 'Alex_Robbinson')
         self.assertEqual(path, ['Hannah_Mullard', 'David_Stone', 'Catherine_Stevens', 'Benjamin_Walker', 'Alex_Robbinson'])
         self.assertEqual(expand.expand_count, 4)

    def test13(self):
         expand.expand_count = 0
         path = sc.a_star_search(dis_mapM, time_mapM, 'Lakshmi_Raman', 'Heather_Raymond')
         self.assertEqual(path, ['Lakshmi_Raman', 'Peter_Kilshaw', 'Olivia_Keeton', 'Heather_Raymond'])
         self.assertEqual(expand.expand_count, 4)

if __name__== "__main__": unittest.main()
