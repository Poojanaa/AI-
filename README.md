**AI:-**<br>
**1.Write a python program to find square root**<br>
number = int(input("enter a number: "))<br>
sqrt = number ** 0.5<br>
print("square root:", sqrt)<br>
**Output:-**<br>
enter a number: 100<br>
square root: 10.0<br>
**2.Python program to solve quadratic equations**<br>
import cmath<br>
x = 1<br>
y = 4<br>
z = 5<br>
w = (y**2) - (4*x*z)<br>
sol1 = (-y-cmath.sqrt(w))/(2*x)<br>
sol2 = (-y+cmath.sqrt(w))/(2*x)<br>
print('The solution are {0} and {1}'.format(sol1,sol2))<br>
**Output:-**<br>
The solution are (-2-1j) and (-2+1j)<br>
**3.Program to swap two variables using python**<br>
x = 5<br>
y = 16<br>
temp = x<br>
x = y<br>
y = temp<br>
print('The value of x after swapping: {}'.format(x))<br>
print('The value of y after swapping: {}'.format(y))<br>
**Output:-**<br>
The value of x after swapping: 16<br>
The value of y after swapping: 5<br>
**4.python program to check armstrong number**<br>
num = int(input("Enter a number: "))<br>
sum = 0<br>
temp = num<br>
while temp > 0<br>:
   digit = temp % 10<br>
   sum += digit ** 3<br>
   temp //= 10<br>
if num == sum:<br>
   print(num,"is an Armstrong number")<br>
else:<br>
   print(num,"is not an Armstrong number")<br>
 **Output:-**<br>
 Enter a number: 2<br>
2 is not an Armstrong number<br>
Enter a number: 1<br>
1 is an Armstrong number<br>
**5.find GCD and HCF**<br>
def compute_hcf(x, y):<br>
    if x > y:<br>
        smaller = y<b>
    else:<br>
        smaller = x<br>
    for i in range(1, smaller+1):<br>
        if((x % i == 0) and (y % i == 0)):<br>
            hcf = i <br>
    return hcf<br>
num1 = 54 <br>
num2 = 24<br>
print("The H.C.F. is", compute_hcf(num1, num2))<br>
**Output:-**<br>
The H.C.F. is 6<br>
**6.find LCM**<br>
def compute_lcm(x, y):<br>
    if x > y:<br>
       greater = x<br>
    else:<br>
       greater = y<br>

    while(True):<br>
       if((greater % x == 0) and (greater % y == 0)):<br>
           lcm = greater<br>
           break<br>
       greater += 1<br>
    return lcm<br>
num1 = 54<br>
num2 = 24<br>
print("The L.C.M. is", compute_lcm(num1, num2))<br>
**Output:-**<br>
The L.C.M. is 216<br>
**7.add two two matrix**<br>
X = [[12,7,3],<br>
    [4 ,5,6],<br>
    [7 ,8,9]]<br>

Y = [[5,8,1],<br>
    [6,7,3],<br>
    [4,5,9]]<br>

result = [[0,0,0],<br>
         [0,0,0],<br>
         [0,0,0]]<br>
for i in range(len(X)):<br>
    for j in range(len(X[0])):<br>
       result[i][j] = X[i][j] + Y[i][j]<br>

for r in result:<br>
   print(r)<br>
**output:-**
[17, 15, 4]<br>
[10, 12, 9]<br>
[11, 13, 18]<br>
<br>
<br>
**8.covert a string data and time**<br>
from datetime import datetime<br>

my_date_string = "Mar 11 2011 11:31AM"<br>

datetime_object = datetime.strptime(my_date_string, '%b %d %Y %I:%M%p')<br>

print(type(datetime_object))<br>
print(datetime_object)<br>
**Output:-**<br>
<class 'datetime.datetime'><br>
2011-03-11 11:31:00<br>
<br>
**9.apend a file**<br>
file1 = open("myfile.txt", "w")<br>
L = ["This is Delhi \n", "This is Paris \n", "This is London"]<br>
file1.writelines(L)<br>
file1.close()<br>
 
# Append-adds at last<br>
file1 = open("myfile.txt", "a")  # append mode<br>
file1.write("Today \n")<br>
file1.close()<br>
 
file1 = open("myfile.txt", "r")<br>
print("Output of Readlines after appending")<br>
print(file1.read())<br>
print()<br>
file1.close()<br>
 
# Write-Overwrites<br>
file1 = open("myfile.txt", "w")  # write mode<br>
file1.write("Tomorrow \n")<br>
file1.close()<br>
 
file1 = open("myfile.txt", "r")<br>
print("Output of Readlines after writing")<br>
print(file1.read())<br>
print()<br>
file1.close()<br>
**Output:-**<br>
Output of Readlines after appending<br>
This is Delhi <br>
This is Paris <br>
This is LondonToday <br>


Output of Readlines after writing<br>
Tomorrow <br>
**10.reverse a number**<br>
num = 1234<br>
reversed_num = 0<br>

while num != 0:<br>
    digit = num % 10<br>
    reversed_num = reversed_num * 10 + digit<br>
    num //= 10<br>

print("Reversed Number: " + str(reversed_num))<br>
**output:-***<br>
Reversed Number: 4321<br>
**11.program to compute a power of a number.**<br>
<br>
<br>
<br>
<br>
<br>
<br>
**Exercises programs.**<br>
**PART-A**<br>
**1.Write a program to implement breadth first search using python.**<br>
graph = {<br>
    '1' : ['2','10'],<br>
    '2' : ['3','8'],<br>
    '3' : ['4'],<br>
    '4' : ['5','6','7'],<br>
    '5' : [],<br>
    '6' : [],<br>
    '7' : [],<br>
    '8' : ['9'],<br>
    '9' : [],<br>
    '10' : []<br>
    }<br>
visited = []<br>
queue = []<br>
def bfs(visited, graph, node):<br>
    visited.append(node)<br>
    queue.append(node)<br>
    while queue:<br>
        m = queue.pop(0)<br>
        print (m, end = " ")<br>
        for neighbour in graph[m]:<br>
            if neighbour not in visited:<br>
                visited.append(neighbour)<br>
                queue.append(neighbour)<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')<br>
**Output:-**<br>
 Following is the Breadth-First Search<br>
1 2 10 3 8 4 9 5 6 7 <br>
**2.write a program to implement a deapth first search using python.**<br>
graph = {<br>
'5' : ['3','7'],<br>
'3' : ['2', '4'],<br>
'7' : ['6'],<br>
'6': [],<br>
'2' : ['1'],<br>
'1':[],<br>
'4' : ['8'],<br>
'8' : []<br>
}<br>
visited = set() <br>
def dfs(visited, graph, node):<br>
    if node not in visited:<br>
        print (node)<br>
    visited.add(node)<br>
    for neighbour in graph[node]:<br>
        dfs(visited, graph, neighbour)<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>
   **Output:-**<br>
   Following is the Depth-First Search<br>
5<br>
3<br>
2<br>
1<br>
4<br>
8<br>
7<br>
6<br>
   <br>
**3.write a program to implement water jug problem using python.**<br>
from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2): <br>
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
        print(amt1, amt2)<br>
        return True<br>
    if visited[(amt1, amt2)] == False:<br>
        print(amt1, amt2)<br>
        visited[(amt1, amt2)] = True<br>
        return (waterJugSolver(0, amt2) or<br>
                waterJugSolver(amt1, 0) or<br>
                waterJugSolver(jug1, amt2) or<br>
                waterJugSolver(amt1, jug2) or<br>
                waterJugSolver(amt1 + min(amt2, (jug1-amt1)),<br>
                amt2 - min(amt2, (jug1-amt1))) or<br>
                waterJugSolver(amt1 - min(amt1, (jug2-amt2)),<br>
                amt2 + min(amt1, (jug2-amt2))))<br>
    else:<br>
        return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>
<br>
**Output:-**<br>
Steps: <br>
0 0<br>
4 0<br>
4 3<br>
0 3<br>
3 0<br>
3 3<br>
4 2<br>
0 2<br>
True<br>
<br>
**4.write a program to implement tower of hannoi using python.**<br>
def TowerOfHanoi(n , source, destination, auxiliary):<br>
    if n==1:<br>
        print ("Move disk 1 from source",source,"to destination",destination)<br>
        return<br>
    TowerOfHanoi(n-1, source, auxiliary, destination)<br>
    print ("Move disk",n,"from source",source,"to destination",destination)<br>
    TowerOfHanoi(n-1, auxiliary, destination, source)<br>
n = 4<br>
TowerOfHanoi(n,'A','B','C')<br>
<br>
**Output:-**<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 3 from source A to destination C<br>
Move disk 1 from source B to destination A<br>
Move disk 2 from source B to destination C<br>
Move disk 1 from source A to destination C<br>
Move disk 4 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 2 from source C to destination A<br>
Move disk 1 from source B to destination A<br>
Move disk 3 from source C to destination B<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
<br>
**5.write a program to implement best first search using python.**<br>
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>
<br>
# for implementing BFS | returns path having lowest cost<br>
def best_first_search(source, target, n):<br>
    visited = [0] * n<br>
    visited[source] = True<br>
    pq = PriorityQueue()<br>
    pq.put((0, source))<br>
    while pq.empty() == False:<br>
        u = pq.get()[1]<br>
        print(u, end=" ") # the path having lowest cost<br>
        if u == target:<br>
            break<br>
        for v, c in graph[u]:<br>
            if visited[v] == False:<br>
                visited[v] = True<br>
                pq.put((c, v))<br>
    print()<br>
    <br>
# for adding edges to graph<br>
def addedge(x, y, cost):<br>
    graph[x].append((y, cost))<br>
    graph[y].append((x, cost))<br>
    
v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>
e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
    x, y, z = list(map(int, input().split()))<br>
    addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("\nPath: ", end = "")<br>
best_first_search(source, target, v)<br>
<br>
**Output:-**<br>
Enter the number of nodes: 4<br>
Enter the number of edges: 5<br>
Enter the edges along with their weights:<br>
0 1 1<br>
0 2 1<br>
0 3 2<br>
2 3 2<br>
1 3 3<br>
Enter the Source Node: 2<br.
Enter the Target/Destination Node: 1<br>
<br>
Path: 2 0 1 <br>
<br>
**6.Write a Program to Implement Tic-Tac-Toe application using Python.**<br>
import numpy as np<br>
import random<br>
from time import sleep<br>

def create_board():<br>
    return(np.array([[0, 0, 0],<br>
              [0, 0, 0],<br>
            [0, 0, 0]]))<br>

def possibilities(board):<br>
    l = []<br>
<br>
    for i in range(len(board)):<br>
        for j in range(len(board)):<br>
<br>
            if board[i][j] == 0:<br>
                l.append((i, j))<br>
    return(l)<br>

def random_place(board, player):<br>
    selection = possibilities(board)<br>
    current_loc = random.choice(selection)<br>
    board[current_loc] = player<br>
    return(board)<br>

def row_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>

        for y in range(len(board)):<br>
            if board[x, y] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br>
    return(win)<br>

def col_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>

        for y in range(len(board)):<br>
            if board[y][x] != player:<br>
                win = False<br>
                continue<br>

        if win == True:<br>
            return(win)<br>
    return(win)<br>

def diag_win(board, player):<br>
    win = True<br>
    y = 0<br>
    for x in range(len(board)):<br>
        if board[x, x] != player:<br>
            win = False<br>
    if win:<br>
        return win<br>
    win = True<br>
    if win:<br>
        for x in range(len(board)):<br>
            y = len(board) - 1 - x<br>
            if board[x, y] != player:<br>
                win = False<br>
    return win<br>
<br>
def evaluate(board):<br>
    winner = 0<br>

    for player in [1, 2]:<br>
        if (row_win(board, player) or<br>
            col_win(board,player) or<br>
            diag_win(board,player)):<br>
            winner = player<br>
    if np.all(board != 0) and winner == 0:<br>
            winner = -1<br>
            <br>
    return winner<br>
def play_game():<br>
    board, winner, counter = create_board(), 0, 1<br>
    print(board)<br>
    sleep(2)<br>

    while winner == 0:<br>
        for player in [1, 2]:<br>
            board = random_place(board, player)<br>
            print("Board after " + str(counter) + " move")<br>
            print(board)<br>
            sleep(2)<br>
            counter += 1<br>
            winner = evaluate(board)<br>
            if winner != 0:<br>
                break<br>
    return(winner)<br>
print("Winner is: " + str(play_game()))<br>
<br>
**Output:-**<br>
[[0 0 0]<br>
 [0 0 0]<br>
 [0 0 0]]<br>
Board after 1move<br>
[[0 0 0]<br>
 [0 1 0]<br>
 [0 0 0]]<br>
Board after 2 move<br>
[[0 0 0]<br>
 [0 1 0]<br>
 [2 0 0]]<br>
Board after 3 move<br>
[[0 0 0]<br>
 [0 1 0]<br>
 [2 1 0]]<br>
Board after 4 move<br>
[[0 0 2]<br>
 [0 1 0]<br>
 [2 1 0]]<br>
Board after 5 move<br>
[[0 0 2]<br>
 [0 1 1]<br>
 [2 1 0]]<br>
Board after 6 move<br>
[[2 0 2]<br>
 [0 1 1]<br>
 [2 1 0]]<br>
Board after 7 move<br>
[[2 1 2]<br>
 [0 1 1]<br>
 [2 1 0]]<br>
Winner is: 1<br>
**7.Write a Program to Implement 8-Puzzle Problem using Python.**<br>
import copy<br>
from heapq import heappush, heappop<br>
n = 3<br>
row = [ 1, 0, -1, 0 ]<br>
col = [ 0, -1, 0, 1 ]<br>

class priorityQueue:<br>
    def __init__(self):<br>
        self.heap = []<br>
    def push(self, k):<br>
         heappush(self.heap, k)<br>
    def pop(self):<br>
        return heappop(self.heap)<br>
    def empty(self):<br>
        if not self.heap:<br>
            return True<br>
        else:<br>
            return False<br>
<br>
class node:<br>
        def __init__(self, parent, mat, empty_tile_pos,cost, level):<br>
            self.parent = parent<br>
            self.mat = mat<br>
            self.empty_tile_pos = empty_tile_pos<br>
            self.cost = cost<br>
            self.level = level<br>
            
        def __lt__(self, nxt):<br>
            return self.cost < nxt.cost<br>
def calculateCost(mat, final) -> int:<br>
    count = 0<br>
    for i in range(n):<br>
        for j in range(n):<br>
            if ((mat[i][j]) and (mat[i][j] != final[i][j])):<br>
                count += 1<br>
    return count<br>

def newNode(mat, empty_tile_pos, new_empty_tile_pos,level, parent, final) -> node:<br>
    new_mat = copy.deepcopy(mat)<br>
    x1 = empty_tile_pos[0]<br>
    y1 = empty_tile_pos[1]<br>
    x2 = new_empty_tile_pos[0]<br>
    y2 = new_empty_tile_pos[1]<br>
    new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]<br>
    cost = calculateCost(new_mat, final)<br>
    new_node = node(parent, new_mat, new_empty_tile_pos,cost, level)<br>
    return new_node<br>

def printMatrix(mat):<br>
    for i in range(n):<br>
        for j in range(n):<br>
            print("%d " % (mat[i][j]), end = " ")<br>
        print()<br>

def isSafe(x, y):<br>
    return x >= 0 and x < n and y >= 0 and y < n<br>

def printPath(root):<br>
    if root == None:<br>
        return<br>
    printPath(root.parent)<br>
    printMatrix(root.mat)<br>
    print()<br>
    <br>
def solve(initial, empty_tile_pos, final):<br>
    pq = priorityQueue()<br>
    cost = calculateCost(initial, final)<br>
    root = node(None, initial,empty_tile_pos, cost, 0)<br>
    pq.push(root)<br>
    while not pq.empty():<br>
        minimum = pq.pop()<br>
        if minimum.cost == 0:<br>
            printPath(minimum)<br>
            return<br>
        for i in range(n):<br>
            new_tile_pos = [minimum.empty_tile_pos[0] + row[i],minimum.empty_tile_pos[1] + col[i], ]<br>
            if isSafe(new_tile_pos[0], new_tile_pos[1]):<br>
                child=newNode(minimum.mat,minimum.empty_tile_pos,new_tile_pos,minimum.level+1,minimum,final,)<br>
                pq.push(child)<br>
initial = [ [ 1, 2, 3 ],[ 5, 6, 0 ],[ 7, 8, 4 ] ]<br>
final = [ [ 1, 2, 3 ],[ 5, 8, 6 ],[ 0, 7, 4 ] ]<br>
empty_tile_pos = [ 1, 2 ]<br>
solve(initial, empty_tile_pos, final)<br>
<br>
**Output:-**<br>
1  2  3  <br>
5  6  0  <br>
7  8  4  <br>
<br>
1  2  3  <br>
5  0  6  <br>
7  8  4  <br>
<br>
1  2  3  <br>
5  8  6  <br>
7  0  4  <br>
<br>
1  2  3  <br>
5  8  6  <br>
0  7  4  <br>
<br>
**8.Write a Program to Implement Travelling Salesman problem using Python.<br>
   from sys import maxsize<br>
from itertools import permutations<br>
V = 4<br>
def travellingSalesmanProblem(graph, s):<br>
    vertex = []<br>
    for i in range(V):<br>
        if i != s:<br>
            vertex.append(i)<br>
    min_path = maxsize<br>
    next_permutation=permutations(vertex)<br>
    for i in next_permutation:<br>
        current_pathweight = 0<br>
        k = s<br>
        for j in i:<br>
            current_pathweight += graph[k][j]<br>
            k = j<br>
        current_pathweight += graph[k][s]<br>
        min_path = min(min_path, current_pathweight)<br>
    return min_path<br>
if __name__ == "__main__":<br>
    graph = [[0, 10, 15, 20], [10, 0, 35, 25],<br>
        [15, 35, 0, 30], [20, 25, 30, 0]]<br>
    s = 0<br>
    print(travellingSalesmanProblem(graph, s))<br>
<br>
**Output:-**<br>
80
<br>
**9.Write a program to implement the FIND-S Algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file.**<br>

   
