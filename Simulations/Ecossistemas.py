# %% Progam goal
'''
Simulation of a simple ecosystem comprised of plants, herbivores and carnivores. 


by: 
    Diogo Durão 55739
    Mónica Dyreby 55808

'''

# %% Imports

%matplotlib qt
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import copy
import matplotlib.patches as mpatches

# %% Classes


class Void():
    def __init__(self, pos):
        self.position = pos

    def position(self):
        return self.position


class Creature():  # this is a superclass
    def __init__(self, pos, strg):
        self.position = pos
        self.strength = strg

    def position(self):
        return self.position

    def strength(self):
        return self.strength

    def upd_pos(self, new_pos):
        self.position = new_pos
        return self.position

    def upd_strg(self, new_str):
        self.strength = new_str
        return self.strength


class Herbivore(Creature):  # this is subclass

    def eat(self, fields):
        ''' Function's goal: describes herbivore eating patterns     
            Function's variables:
                fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
            Function's returns: 
                fields: updated fields      
        '''

        x = self.position[0]
        y = self.position[1]

        veggies = neighbours_of_class(x, y, fields, Plant)

        if len(veggies) != 0:  # there are plants next to the herbivore
            eating_time(self, fields, veggies, x, y, 0)

        else:  # there are no plants next to the herbivore
            if self.strength > 0:
                move(self, fields, x, y, 0)
            else:
                starve(self, fields, x, y, 0)

        return fields


class Carnivore(Creature):  # this is subclass

    def eat(self, fields):
        ''' Function's goal: describes carnivore eating patterns     
            Function's variables:
                fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
            Function's returns: 
                fields: updated fields      
        '''

        x = self.position[0]
        y = self.position[1]

        meaties = neighbours_of_class(x, y, fields, Herbivore)

        if len(meaties) != 0:  # there are plants next to the herbivore
            eating_time(self, fields, meaties, x, y, 1)

        else:  # there are no plants next to the herbivore
            if self.strength > 0:
                move(self, fields, x, y, 1)
            else:
                starve(self, fields, x, y, 1)

        return fields


class Plant(Creature):  # this is subclass
    None


# %% Initilization
def initiate(x_matrix, y_matrix):
    ''' Function's goal: Inicialize a matrix to simulate an ecosystem. 
        Function's variables: 
            x_matrix, y_matrix: matrix size parameters    
        Function's returns: 
            ob_field: matrix comprised of all the objects that represent creatures in the ecosystem. 
            h, c, p, v: lists of class specific objects (list of herbivores, carnivores, plants and voids)
            x_matrix y_matrix 
            colour_field: matrix comprised of float numbers that represents a creature's class and strength
    '''
    classes_tab = ["H", "C", "P"]
    st_field = np.random.choice(
        classes_tab, (x_matrix, y_matrix), p=[3/13, 1/13, 9/13])

    h = []
    c = []
    p = []
    v = []
    ob_field = np.zeros((x_matrix, y_matrix), dtype="object_")
    colour_field = np.zeros((x_matrix, y_matrix), dtype="float")

    for i in range(0, x_matrix):
        for j in range(0, y_matrix):
            if st_field[i, j] == 'H':
                herb = Herbivore((i, j), rand.randint(0, 2))
                h.append(herb)
                ob_field[i, j] = herb

            elif st_field[i, j] == 'C':
                carn = Carnivore((i, j), rand.randint(0, 2))
                c.append(carn)
                ob_field[i, j] = carn

            else:
                plant = Plant((i, j), rand.randint(0, 2))
                p.append(plant)
                ob_field[i, j] = plant

    update_colour_field(colour_field, ob_field)

    return ob_field, h, c, p, v, x_matrix, y_matrix, colour_field


def initiate_no_carnivores(x_matrix, y_matrix):
    ''' Function's goal: Inicialize a matrix to simulate an ecosystem without carnivores. 
        Function's variables:
            x_matrix, y_matrix :matrix size parameters    
        Function's returns: 
            ob_field: matrix comprised of all the objects that represent creatures in the ecosystem. 
            h, c, p, v: lists of class specific objects (list of herbivores, carnivores, plants and voids)
            x_matrix y_matrix 
            colour_field: matrix comprised of float numbers that represent creatures 
    '''

    classes_tab = ["H", "P"]
    st_field = np.random.choice(
        classes_tab, (x_matrix, y_matrix), p=[3/12, 9/12])

    h = []
    p = []
    c = []  # does nothing - exists to mantain the order of returned objects
    v = []
    ob_field = np.zeros((x_matrix, y_matrix), dtype="object_")
    colour_field = np.zeros((x_matrix, y_matrix), dtype="float")

    for i in range(0, x_matrix):
        for j in range(0, y_matrix):
            if st_field[i, j] == 'H':
                herb = Herbivore((i, j), rand.randint(0, 2))
                h.append(herb)
                ob_field[i, j] = herb

            else:
                plant = Plant((i, j), rand.randint(0, 2))
                p.append(plant)
                ob_field[i, j] = plant

    update_colour_field(colour_field, ob_field)

    return ob_field, h, c, p, v, x_matrix, y_matrix, colour_field


# %% Neighbours functions

def neighbours(x, y, fields):
    ''' Function's goal: group the four von Neumann neighbours of a matrix element/creature  
        Function's variables:
            x, y: element's position in matrix 
            fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
        Function's returns: 
            nb: list of objects that comprise the four neighbours of the element in (x,y) position
    '''

    obj_field = fields[0]
    x_matrix = fields[5]
    y_matrix = fields[6]

    if x < x_matrix-1 and y < y_matrix-1:
        nb = [obj_field[x-1, y], obj_field[x+1, y],
              obj_field[x, y-1], obj_field[x, y+1]]
    elif x == x_matrix-1 and y < y_matrix-1:
        nb = [obj_field[x-1, y], obj_field[0, y],
              obj_field[x, y-1], obj_field[x, y+1]]
    elif x < x_matrix-1 and y == y_matrix-1:
        nb = [obj_field[x-1, y], obj_field[x+1, y],
              obj_field[x, y-1], obj_field[x, 0]]
    else:
        nb = [obj_field[x-1, y], obj_field[0, y],
              obj_field[x, y-1], obj_field[x, 0]]

    return nb


def neighbours_of_class(x, y, fields, class_is):
    ''' Function's goal: create randomly ordered list with neighbours of a certain class
        Function's variables:
            x, y: element's position in matrix 
            fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
            class_is: type of class (Plant, Herbivore, Carnivore, Void) this class specific neighbour list will be comprised of  
        Function's returns: 
            nbc: randomized list of objects that comprise the class specific neighbour list of the element in (x,y) position
    '''

    nb_list = neighbours(x, y, fields)
    nbc = []
    for i in range(0, 4):
        if isinstance(nb_list[i], class_is):
            nbc.append(nb_list[i])

    rand.shuffle(nbc)  # disarrange list
    return nbc

# %% Class functions


def eating_time(obj, fields, foodies, x, y, mode):
    ''' Function's goal: general function used to describe the eating process of herbivores or plants by carnivores or herbivores respectively   
        Function's variables:
            obj: current "eater" creature 
            fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
            x, y: "eater" element's position in matrix 
            foodies: list comprised of all edible neighbour creatures for the "eater" in question 
            mode: int variable that specifies "eater" class (i.e if mode is 0 the "eater" is a herbivore)
        Function's returns: 
            No returns 
            Creature strength is automatically updated in all the lists 
             because these objects, on the object field, share the same memory adress as the ones in the lists             
    '''

    obj_field = fields[0]
    h_list = fields[1]
    c_list = fields[2]
    p_list = fields[3]
    v_list = fields[4]

    if mode == 0:  # eater is herbivore
        eater_list = h_list
        food_list = p_list
        class_eater = Herbivore
        class_walk = (Plant, Void)

    else:  # eater is carnivore
        eater_list = c_list
        food_list = h_list
        class_eater = Carnivore
        class_walk = (Herbivore, Plant, Void)

    new_str = foodies[0].strength - 1
    f_pos = foodies[0].position
    if(new_str == -1):  # food has died - replace with void
        obj_field[f_pos] = Void(f_pos)
        v_list.append(obj_field[f_pos])
        food_list.remove(foodies[0])

    else:  # food loses strength
        obj_field[f_pos].upd_strg(new_str)

    o_str = obj.strength
    o_position = obj.position
    if o_str < 2:  # the eater is weak or normal - gains strength
        new_str = o_str + 1
        obj_field[o_position].upd_strg(new_str)

    else:  # eater is strong
        place_of_birth = neighbours_of_class(x, y, fields, class_walk)[0]
        position_of_birth = place_of_birth.position
        newborn = class_eater(position_of_birth, 0)
        obj_field[position_of_birth] = newborn
        eater_list.append(obj_field[position_of_birth])

        if isinstance(place_of_birth, Void):
            v_list.remove(place_of_birth)
        elif isinstance(place_of_birth, Plant):
            p_list.remove(place_of_birth)
        elif isinstance(place_of_birth, Herbivore):
            h_list.remove(place_of_birth)


def move(obj, fields, x, y, mode):
    ''' Function's goal: general function used to describe the movement of herbivores or carnivores  
        Function's variables:
            obj: current "eater" creature 
            fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
            x, y: "eater" element's position in matrix 
            mode: int variable that specifies "eater" class (i.e if mode is 0 the "eater" is a herbivore)
        Function's returns: 
            No returns 
            Creature position and strength are automatically updated in all the lists 
             because these objects, on the object field, share the same memory adress as the ones in the lists             
    '''
    obj_field = fields[0]
    h_list = fields[1]
    c_list = fields[2]
    p_list = fields[3]
    v_list = fields[4]

    if mode == 0:  # eater is herbivore
        eater_list = h_list
        class_eater = Herbivore
        free_spaces = neighbours_of_class(x, y, fields, Void)

    else:  # eater is carnivore
        eater_list = c_list
        class_eater = Carnivore
        free_spaces = neighbours_of_class(x, y, fields, (Void, Plant))

    new_str = obj.strength - 1
    o_position = obj.position

    if (len(free_spaces) != 0):  # has space to walk
        free_space = free_spaces[0]
        new_position = free_space.position

        if isinstance(free_space, Void):  # updates the new location
            v_list.remove(free_space)
        elif isinstance(free_space, Plant):
            p_list.remove(free_space)
        obj_field[new_position] = class_eater(new_position, new_str)
        eater_list.append(obj_field[new_position])

        eater_list.remove(obj_field[o_position])  # updates the old location
        obj_field[o_position] = Void(o_position)
        v_list.append(obj_field[o_position])

    else:  # no space to walk
        obj_field[o_position].upd_strg(new_str)


def starve(obj, fields, x, y, mode):
    ''' Function's goal: general function used to describe the behaviour of herbivores or carnivores when their neighbour list has no edible creatures  
        Function's variables:
            obj: current "eater" creature 
            fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
            x, y: "eater" element's position in matrix 
            mode: int variable that specifies "eater" class (i.e if mode is 0 the "eater" is a herbivore)
        Function's returns: 
            No returns 
            Creature position is automatically updated in all the lists 
             because these objects, on the object field, share the same memory adress as the ones in the lists             
    '''

    obj_field = fields[0]
    h_list = fields[1]
    c_list = fields[2]
    v_list = fields[4]

    if mode == 0:  # eater is herbivore
        eater_list = h_list
    else:  # eater is carnivore
        eater_list = c_list

    o_position = obj.position
    eater_list.remove(obj_field[o_position])
    obj_field[o_position] = Void(o_position)
    v_list.append(obj_field[o_position])


# %% Iterations Functions
'''
 fields: variable containing information regarding all aspects of the matrix/ecosystem (e.g objects, plant list...) 
 
'''


def all_herbivores(fields):
    ''' Function's goal: all herbivores will eat 
        No returns 
    '''

    h_list = fields[1]
    h_tuple = tuple(h_list)

    for i in range(0, len(h_tuple)):
        h_tuple[i].eat(fields)


def all_carnivores(fields):
    ''' Function's goal: all carnivores will eat 
        No returns 
    '''

    c_list = fields[2]
    c_tuple = tuple(c_list)

    for i in range(0, len(c_tuple)):
        c_tuple[i].eat(fields)


def all_plants_grow(fields):
    ''' Function's goal: increases strength of all plants 
        No returns               
    '''

    p_list = fields[3]

    for i in range(0, len(p_list)):
        p_strength = p_list[i].strength
        if p_strength < 2:
            p_list[i].upd_strg(p_strength + 1)


def plant_genesis(fields):
    ''' Function's goal: replace all voids with plants
        No returns               
    '''
    obj_field = fields[0]
    p_list = fields[3]
    v_list = fields[4]
    v_tuple = tuple(v_list)

    for i in range(0, len(v_tuple)):
        v_pos = v_tuple[i].position
        v_list.remove(v_tuple[i])
        obj_field[v_pos] = Plant(v_pos, 0)
        p_list.append(obj_field[v_pos])


def life_goes(fields):
    ''' Function's goal: run the four previous funtions in the specified order
        No returns               
    '''

    all_herbivores(fields)
    all_carnivores(fields)
    all_plants_grow(fields)
    plant_genesis(fields)

# %% Update Field Function


def update_colour_field(colour_field, obj_field):
    ''' Function's goal: update the colour_ field matrix's elements    
        Function's variables:
            colour_field: field of float numbers that describe the creature's class and strength 
            ob_field: matrix comprised of all the objects that represent creatures in the ecosystem
        Function's returns: 
            No returns                     
    '''

    x = obj_field.shape[0]
    y = obj_field.shape[1]

    for i in range(0, x):
        for j in range(0, y):
            if isinstance(obj_field[i, j], Herbivore):
                if obj_field[i, j].strength == 0:
                    colour = 1.2
                elif obj_field[i, j].strength == 1:
                    colour = 1.1
                else:
                    colour = 1
            elif isinstance(obj_field[i, j], Carnivore):
                if obj_field[i, j].strength == 0:
                    colour = 4.2
                elif obj_field[i, j].strength == 1:
                    colour = 4.1
                else:
                    colour = 4

            elif isinstance(obj_field[i, j], Plant):
                if obj_field[i, j].strength == 0:
                    colour = 7.2
                elif obj_field[i, j].strength == 1:
                    colour = 7.1
                else:
                    colour = 7
            else:
                colour = 9
            colour_field[i, j] = colour

# %% Main Function


def circle_of_life(N, x, y, mode):
    ''' Function's goal: run the program N times   
        Function's variables:
            N: number of iteration 
            ob_field: matrix comprised of all the objects that represent creatures in the ecosystem
            x, y: matrix size parameters 
            mode: int variable that indicates simulation type (i.e if mode is 0 the simulation has carnivores, else it has no carnivores)
        Function's returns: 
            N_tuple: list all all iterations
            all_percentages: list with statistical information for each iteration 
            save: list with the colour_field for each iteration                        
    '''

    if mode == 0:  # with carnivores
        fields = initiate(x, y)
    else:  # without carnivores
        fields = initiate_no_carnivores(x, y)

    lists = initiate_plot_lists(N)

    save = []
    save = copy.deepcopy([fields[7]])  # saves the current colour_field

    for i in range(1, N + 1):
        life_goes(fields)
        lists = plot_lists(N, x, y, fields, lists, i, mode)
        update_colour_field(fields[7], fields[0])
        save = save + copy.deepcopy([fields[7]])

    N_tuple = lists[0]
    all_percentages = lists[5]

    return N_tuple, all_percentages, save

# %% Auxiliary plot funtions


def initiate_plot_lists(N):
    ''' Function's goal: inicialize lists for statistical purposes   
        Function's variables:
            N: number of iterations 
        Function's returns: 
            N_tuple: list of all iterations
            several empty prepared lists that will be used in other functions     
    '''

    N_tuple = range(1, N + 1)

    # 0 is herbivore, 1 is carnivore and 2 is plant
    creatures_perc = [[], [], []]

    class_h_perc = [[], [], []]  # 0 is weak, 1 is normal and 2 is strong
    class_c_perc = [[], [], []]
    class_p_perc = [[], [], []]

    all_perc = [[], [], []]

    return N_tuple, creatures_perc, class_h_perc, class_c_perc, class_p_perc, all_perc


def strength_graph(class_list, class_percentages):
    ''' Function's goal: categorize the strengths of a specific class  
        Function's variables:
            class_list: class whose strength will be categorized
            class_percentages: list of percentages for each class (inicially empty)
        Function's returns: 
            class_percentages: updated percentages      
    '''

    lis_len = len(class_list)
    weak = 0
    normal = 0
    strong = 0

    for i in range(0, lis_len):
        if class_list[i].strength == 0:
            weak += 1
        elif class_list[i].strength == 1:
            normal += 1
        else:
            strong += 1

    class_percentages[0].append(weak*100/(lis_len))
    class_percentages[1].append(normal*100/(lis_len))
    class_percentages[2].append(strong*100/(lis_len))


    return class_percentages


def plot_lists(N, x_matrix, y_matrix, fields, lists, i, mode):
    ''' Function's goal: update the empty lists created for statistical purposes   
        Function's variables:
            x_matrix, y_matrix, fields, 
            lists: lists created for statistical purposes (inicially empty)
            i: int value that indicates the iteration number (iteration number is not the number of iterations)
            mode: int value that determines if the carnivores are a part of the simulation 
              (i.e. mode == 0 means that they are, therefore they must be included in the statistical analysis) 
        Function's returns: 
            class_percentages: updated percentages      
    '''

    p_list = fields[3]
    c_list = fields[2]
    h_list = fields[1]
    x = fields[5]
    y = fields[6]

    N_tuple, creatures_perc, class_h_perc, class_c_perc, class_p_perc, all_perc = lists

    # each index will have the percentage of plants in the corresponding iteration
    creatures_perc[0].append(len(h_list)*100/(x*y))
    creatures_perc[2].append(len(p_list)*100/(x*y))

    # information about the percentages of each creature will be placed in a matrix format for better organization
    all_perc[0] = strength_graph(h_list, class_h_perc)
    all_perc[0].append(creatures_perc[0])

    all_perc[2] = strength_graph(p_list, class_p_perc)
    all_perc[2].append(creatures_perc[2])

    if mode == 0:
        creatures_perc[1].append(len(c_list)*100/(x*y))
        all_perc[1] = strength_graph(c_list, class_c_perc)
        all_perc[1].append(creatures_perc[1])

    return N_tuple, creatures_perc, class_h_perc, class_c_perc, class_p_perc, all_perc


# %% Percentage Plots
''' Functions's goal: create the following plots: 
        -percentage of each class in each iteration 
        -percentage of herbivore, carnivore and plant strength in each iteration  
    Function's variables:
         N_tuple: list all all iterations 
         all_percentages: list with statistical information for each iteration 
         figsize_x, figsize_y: size of figure
         mode: int value that determines if the carnivores are a part of the simulation 
              (i.e. mode == 0 means that they are, therefore they must be included in the plots) 
    Function's returns: 
            No returns
'''


def all_creatures_percentages_plot(N_tuple, all_percentages, figsize_x, figsize_y, mode):
    percentages_h, percentages_c, percentages_p = all_percentages

    fig, ax_creature = plt.subplots(figsize=(figsize_x, figsize_y))
    ax_creature.xaxis.set_major_locator(MaxNLocator(integer=True))

    h, = ax_creature.plot(N_tuple, percentages_h[3], color='b', marker='o')
    p, = ax_creature.plot(N_tuple, percentages_p[3], color='g', marker='o')
    if mode == 0:
        c, = ax_creature.plot(N_tuple, percentages_c[3], color='r', marker='o')
        ax_creature.legend([h, c, p], ['Herbivores', 'Carnivores',
                           'Plants'], loc='upper right', shadow=True)
    else:
        ax_creature.legend([h, p], ['Herbivores', 'Plants'],
                           loc='upper right', shadow=True)

    ax_creature.set_title('percentage of creatures vs iterations')
    ax_creature.set_ylabel(r'percentage ')
    ax_creature.set_xlabel(r'iterations')


def herbivore_percentages_plot(N_tuple, all_percentages, figsize_x, figsize_y):
    percentages_h = all_percentages[0]

    fig, ax_strg_h = plt.subplots(figsize=(figsize_x, figsize_y))
    ax_strg_h.xaxis.set_major_locator(MaxNLocator(integer=True))

    w, = ax_strg_h.plot(N_tuple, percentages_h[0], color='y', marker='o')
    n, = ax_strg_h.plot(N_tuple, percentages_h[1], color='brown', marker='o')
    s, = ax_strg_h.plot(N_tuple, percentages_h[2], color='black', marker='o')

    ax_strg_h.legend([w, n, s], ['Weak Herbivores', 'Normal Herbivores',
                     'Strong Herbivores'], loc='upper right', shadow=True)
    ax_strg_h.set_title('Herbivore strength vs iterations')
    ax_strg_h.set_ylabel(r'Herbivore strength')
    ax_strg_h.set_xlabel(r'iterations')


def carnivore_percentages_plot(N_tuple, all_percentages, figsize_x, figsize_y):

    percentages_c = all_percentages[1]

    fig, ax_strg_c = plt.subplots(figsize=(figsize_x, figsize_y))
    ax_strg_c.xaxis.set_major_locator(MaxNLocator(integer=True))
    w, = ax_strg_c.plot(N_tuple, percentages_c[0], color='y', marker='o')
    n, = ax_strg_c.plot(N_tuple, percentages_c[1], color='brown', marker='o')
    s, = ax_strg_c.plot(N_tuple, percentages_c[2], color='black', marker='o')

    ax_strg_c.legend([w, n, s], ['Weak Carnivores', 'Normal Carnivores',
                     'Strong Carnivores'], loc='upper right', shadow=True)
    ax_strg_c.set_title('Carnivore strength vs iterations')
    ax_strg_c.set_ylabel(r'Carnivore strength')
    ax_strg_c.set_xlabel(r'iterations')


def plant_percentages_plot(N_tuple, all_percentages, figsize_x, figsize_y):

    percentages_p = all_percentages[2]

    fig, ax_strg_p = plt.subplots(figsize=(figsize_x, figsize_y))
    ax_strg_p.xaxis.set_major_locator(MaxNLocator(integer=True))
    w, = ax_strg_p.plot(N_tuple, percentages_p[0], color='y', marker='o')
    n, = ax_strg_p.plot(N_tuple, percentages_p[1], color='brown', marker='o')
    s, = ax_strg_p.plot(N_tuple, percentages_p[2], color='black', marker='o')

    ax_strg_p.legend([w, n, s], ['Weak Plants', 'Normal Plants',
                     'Strong Plants'], loc='upper right', shadow=True)
    ax_strg_p.set_title('Plant strength vs iterations')
    ax_strg_p.set_ylabel(r'Plant strength')
    ax_strg_p.set_xlabel(r'iterations')


def all_plots(N_tuple, all_percentages, figsize_x, figsize_y, mode):
    ''' Function's goal: plot the previously mentioned plots,
                        avoiding, if necessary, the plotting of the carnivore strength   
    '''

    all_creatures_percentages_plot(
        N_tuple, all_percentages, figsize_x, figsize_y, mode)
    herbivore_percentages_plot(N_tuple, all_percentages, figsize_x, figsize_y)
    if mode == 0:
        carnivore_percentages_plot(
            N_tuple, all_percentages, figsize_x, figsize_y)
    plant_percentages_plot(N_tuple, all_percentages, figsize_x, figsize_y)


# %% Animation

st = time.time()

'''
To alter the simulation parameters keep in mind the following guidelines: 
    circle_of_life(numero de iterações, x of the matrix, y of the matrix, mode(0 to include carnivores, 1 to exclude the carnivores))
    all_plots(N, A, x - figure size, y - figure size, mode in question)

'''

N, A, save = circle_of_life(500, 50, 50, 0)
all_plots(N, A, 8, 8, 0)


''' Creates the animation of the evolution of the ecosystem 

    To save: uncomment the two commented lines bellow.
             REPLACE "saving location " with the appropriate adress, example included in report 
             
'''


colour_field = save
fig, ax = plt.subplots()
ax.set_title("Animação Ecossistema")

colors1 = plt.cm.Blues_r(np.linspace(0., 1, 128))
colors2 = plt.cm.Reds_r(np.linspace(0, 1, 128))
colors3 = plt.cm.Greens_r(np.linspace(0., 1, 128))
colors = np.vstack((colors1, colors2, colors3))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

values = [1.0, 1.1, 1.2, 4.0, 4.1, 4.2, 7.0, 7.1, 7.2]
labels =["H strong", "H normal" ,"H weak","C strong", "C normal", "C weak", "P strong", "P normal", "P weak" ]
ims = []
for i in range(len(save)):
    im = ax.imshow(colour_field[i], mymap, None, vmin=0, vmax=9)
    if i == 0:
        # show an initial one first
        ax.imshow(colour_field[i], mymap, None, vmin=0, vmax=9, animated=True)
    colors = [im.cmap(im.norm(value)) for value in values]
    
    patches = [mpatches.Patch(color=colors[i], label = labels[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ims.append([im])


ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)


#writergif = animation.PillowWriter(fps=15)
# ani.save(r'"saving location", writer = writergif)

et = time.time()
elapsed_time = et - st
print("Total execution Time:", elapsed_time, "seconds")
