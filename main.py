import collections
from typing import List
from typing import Optional



# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors # if neighbors is not None else []



def rightSideView(root: Optional[TreeNode]) -> List[int]: # Others' solution.Here root type is None or TreeNode
    print('199. Binary Tree Right Side View(Medium)')
    result = []
    if not root:
       return result
    q=collections.deque([root]) #q only store elements layer by layer
    while q:
        length = len(q)# here we wanna fixed the every layer size of elment when every time loop on the first element in each layer
        for i in range(length):
            root=q.popleft()# here we wanna always find the current treenode at root
            if i==length-1:
                result.append(root.val)
            if root.left:
               q.append(root.left)
            if root.right:
               q.append(root.right)
    return result

def averageOfLevels(root: Optional[TreeNode]) -> List[float]:
    print('637. Average of Levels in Binary Tree(Easy)')
    result = []
    if not root:
        return result
    q = collections.deque([root])
    while q:
        length = len(q)
        count = 0
        for i in range(length):
            root = q.popleft()
            count +=root.val
            if root.left:
              q.append(root.left)
            if root.right:
              q.append(root.right)
        result.append(count/length)
    return result

def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    print('102. Binary Tree Level Order Traversal(Medium)')
    result=[]
    if not root:
        return result
    q = collections.deque([root])
    while q:
        length = len(q)
        tmp_list = []
        for i in range(length):
           root = q.popleft()
           tmp_list.append(root.val)
           if root.left:
               q.append(root.left)
           if root.right:
               q.append(root.right)
        result.append(tmp_list)
    return result

def zigzagLevelOrder(root: Optional[TreeNode]) -> List[List[int]]: #之字替换顺序
    print('103. Binary Tree Zigzag Level Order Traversal(Medium)')
    result = []
    if not root:
        return result
    q=collections.deque([root])
    layer_num=0
    while q:
        length = len(q)
        tmp_list = []
        if layer_num%2==0:
          for i in range(length):
            root=q.popleft()
            tmp_list.append(root.val)
            if root.left:
                q.append(root.left)
            if root.right:
                q.append(root.right)
        else:
          for i in range(length):
            root = q.pop()
            tmp_list.append(root.val)
            if root.right:
                q.appendleft(root.right)
            if root.left:
                q.appendleft(root.left)
        result.append(tmp_list)
        layer_num = layer_num+1
    return result

def getMinimumDifference(root: Optional[TreeNode]) -> int: #my solution is wrong, In general, two nodes cannot have the same value in the binary search tree
    print('530. Minimum Absolute Difference in BST(Easy)')
    node_list = []
    q = collections.deque([])
    min = 100000
    q.append(root)
    if root.left:
      while root.left:
        q.append(root.left)
        root = root.left
      node_list.append(root.val)
      q.pop()
    while q:
        current_q=q.pop()
        node_list.append(current_q.val)
        if current_q.right:
            root = current_q.right
            while root.left:
                q.appendleft(root)
                root = root.left
            node_list.append(root.val)
    for i in range(len(node_list)-1):
       if min> node_list[i+1]-node_list[i]:
           min = node_list[i+1]-node_list[i]
    return min

def inorder(inord: List[int], root: TreeNode) -> None:#left->root->right 中序
    '''      543
       384        652
          445        699
    '''
    if root is None:
        return
    inorder(inord, root.left)
    inord.append(root.val)
    inorder(inord, root.right)

def preorder(preord: List[int], root: TreeNode) -> None:#root->left->right 前序
    if root is None:
        return
    preord.append(root.val)
    preorder(preord, root.left)
    preorder(preord, root.right)

def postorder(postord: List[int], root: TreeNode) -> None:#left->right->root 前序
    if root is None:
        return
    postorder(postord, root.left)
    postorder(postord, root.right)
    postord.append(root.val)


def getMinimumDifferenceOthers(root: Optional[TreeNode]) -> int: #Others' solution, In general, two nodes cannot have the same value in the binary search tree
    print('530. Minimum Absolute Difference in BST(Easy)')
    node_list = []
    inorder(node_list, root)
    for i in range(len(node_list)-1):
       if min> node_list[i+1]-node_list[i]:
           min = node_list[i+1]-node_list[i]
    return min

def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    print('230. Kth Smallest Element in a BST(Medium)')
    node_list = []
    inorder(node_list, root)
    return node_list[k-1]

def isValidBST(root: Optional[TreeNode]) -> bool:
    print('98. Validate Binary Search Tree(Medium)')
    status = False
    if not (root.left or root.right):
      return True
    node_list = []
    inorder(node_list, root)
    for i in range(len(node_list)-1):
       if node_list[i+1]<= node_list[i]:
           status = False
           break
       status = True
    return status


# def numIslands(grid: List[List[str]]) -> int:# others' solution was wrong with DFS
#     m = len(grid)
#     n = len(grid[0])
#     visited_set = set()
#     num_islands = 0
#
#     def dfs(grid, i, j):
#         if i<0 or i>=len(grid) or j<0 or j>=len(grid[0]) or grid[i][j] =='0':
#             return
#         grid[i][j] = '-1'
#         dfs(grid, i + 1 , j)
#         dfs(grid, i - 1 , j)
#         dfs(grid, i , j + 1)
#         dfs(grid, i , j - 1)
#
#     for i in range(m):
#       for j in range(n):
#         if grid[i][j]=='1':
#             num_islands +=1
#             dfs(grid, i, j)
#
#     return num_islands

def numIslands(grid: List[List[str]]) -> int:
    # Others' solution  BFS
    # loop on all the posible anti diagonal line from four directions, above, below, left, right
    print('200. Number of Islands(Medium)')
    if not grid:
        return 0

    m = len(grid)
    n = len(grid[0])
    visited_set = set()
    num_islands = 0
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def bfs(i, j):
      q = collections.deque()
      visited_set.add((i, j))
      q.append((i , j))
      while q:
        row, col = q.popleft()
        for dr, dc in directions:
            i, j = row + dr, col + dc
            if (i in range(m) and j in range(n) and grid[i][j] == '1' and (i,j) not in visited_set):
                q.append((i, j))
                visited_set.add((i, j))

    for i in range(m):
      for j in range(n):
        if grid[i][j]=='1' and (i,j) not in visited_set:
            bfs(i, j)
            num_islands +=1

    return num_islands

def solve(board: List[List[str]]) -> None:# others' solution DFS
    print('130. Surrounded Regions(Medium)')

    m = len(board)
    n = len(board[0])

    def captureRegions(i, j):# DFS
      if (i < 0 or i==m or j<0 or j==n or board[i][j]!='O'):
          return
      board[i][j] = 'T'
      captureRegions(i + 1, j)
      captureRegions(i - 1, j)
      captureRegions(i, j+1)
      captureRegions(i, j-1)


    for i in range(m):## Figure out all the border 'O' and corresponding adjacent 'O' and flip it to 'T'
      for j in range(n):
        if (board[i][j]=='O' and (i in [0, m-1] or j in [0, n-1])):
           captureRegions(i,j)

    for i in range(m):  #Flip remaining 'O' to 'X'
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'

    for i in range(m):  #Flip back from 'T' to 'O'
        for j in range(n):
            if board[i][j] == 'T':
                board[i][j] = '0'


def cloneGraph(node: Optional['Node']) -> Optional['Node']:
    print('133. Clone Graph(Medium)')
    if not node:
        return None
    clone_nodes = {}
    clone_nodes[node] =Node(node.val,[]) # Just for clearning up the old reference
    q = collections.deque([node])
    while q:
        current_visit = q.popleft()
        for neighbor in current_visit.neighbors:
          if neighbor not in clone_nodes:
            clone_nodes[neighbor] = Node(neighbor.val, [])
            q.append(neighbor)
          clone_nodes[current_visit].neighbors.append(clone_nodes[neighbor])
    return clone_nodes[node]

def calcEquation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]: # Others' solution
    print('399. Evaluate Division(Medium)')
    graph = collections.defaultdict(list)
    q = collections.deque()
    visited = set()
    for i, (u, v) in enumerate(equations):
        graph[u].append((v, values[i]))
        graph[v].append((u, 1 / values[i]))
    ans = []

    for f, t in queries:
        q.append((f,1))# here 1 is the default value to multiple any value remains the same
        visited.add(f)
        hasAnswer = False

        if f == t:
            if f not in graph:
              ans.append(-1)
            else:
              ans.append(1)
        else:
           while q:
              node, value = q.pop()
              if node == t:
                 ans.append(value)
                 hasAnswer = True

              for neighbor, factor in graph[node]:
                 if neighbor not in visited:
                    q.append((neighbor, value*factor))
                    visited.add(neighbor)
                 else:
                     if neighbor == t:
                         ans.append(factor)
                         hasAnswer = True
           if not hasAnswer:
              ans.append(-1)
    return ans

def calcEquation2(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:  # Others' solution, this one is better
        print('399. Evaluate Division(Medium)')
        graph = collections.defaultdict(list)
        for i, (u, v) in enumerate(equations):
            graph[u].append((v, values[i]))
            graph[v].append((u, 1 / values[i]))
        ans = []

        def bfs(query_src, query_target):
            if query_src not in graph or query_target not in graph:
                return -1
            q = collections.deque()
            visited = set()
            q.append([query_src,1])
            visited.add(query_src)
            while q:
                node , weight = q.popleft()
                if node == query_target:# be found in graph and divided by itself
                    return weight # should be always 1
                for neighbor, factor in graph[node]:
                    if neighbor not in visited:
                        q.append([neighbor,weight*factor])
                        visited.add(neighbor)
            return -1
        for query in queries:
            an = bfs(query[0],query[1])
            ans.append(an)
        return ans

def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool: # my solution time exceed ,check if any circles exist
    print('207. Course Schedule(Medium)')
    graph = collections.defaultdict(set)
    #visited = set()
    for prerequisite in prerequisites:
       graph[prerequisite[1]].add(prerequisite[0])

    def dfs(currentCourse, visited):
       print("currentCourse:",currentCourse)
       if currentCourse in visited:
          return False
       visited.add(currentCourse)
       if currentCourse not in graph:
           return True
       neighborSet = graph[currentCourse]
       for neighbor in neighborSet:
           if not dfs(neighbor, visited):
               return False
           visited.remove(neighbor)

       return True

    for i in range(numCourses):
       visited = set()
       if not dfs(i, visited):
          return False
    return True

def canFinish2(numCourses: int, prerequisites: List[List[int]]) -> bool: # others' solution
    print('207. Course Schedule(Medium)')
    graph = collections.defaultdict(list)

    for v,u in prerequisites:
       graph[u].append(v)

    def dfs(currentCourse, visited):
       if visited[currentCourse]==1:
          return False
       elif visited[currentCourse]==2:
          return True
       else:
          visited[currentCourse]=1
          neighborList = graph[currentCourse]
          for neighbor in neighborList:
              if not dfs(neighbor, visited):
                 return False
          visited[currentCourse] = 2
          return True

    # 0 = Unknown, 1 = visiting, 2 = visited
    visited = [0] * numCourses
    for i in range(numCourses):
       if not dfs(i, visited):
          return False
    return True

def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:# my solution is wrong ,didn't take circle into consideration
    print('210. Course Schedule II(Medium)')
    graph = collections.defaultdict(list)

    for v,u in prerequisites:
       graph[u].append(v)

    def bfs(currentCourse, visited, path):
       if currentCourse in graph:
           if visited[currentCourse] == 0:
               visited[currentCourse] = 1
               path.append(currentCourse)
           neighborList = graph[currentCourse]
           for neighbor in neighborList:
               if visited[neighbor] == 0:
                   path.append(neighbor)
               visited[neighbor] = 1
               print("456")

    # 0 = Unknown, 1 = visiting, 2 = visited
    visited = [0] * numCourses
    path = collections.deque()
    for i in range(numCourses):
       bfs(i, visited, path)
       if visited[numCourses-1]==1:
           break
    for i in range(len(visited)):
        if visited[i]==0:
            path.append(i)

    return list(path)


def findOrder2(numCourses: int, prerequisites: List[List[int]]) -> List[int]:  # others'solution  BFS
    print('210. Course Schedule II(Medium)')
    graph = collections.defaultdict(list)
    in_degree = [0] * numCourses

    for v, u in prerequisites:
        graph[u].append(v)
        in_degree[v] += 1

    path = collections.deque()
    for i in range(numCourses): # place the nodes which do not have no pre-node as higher priority, find the start nodes
        if in_degree[i] == 0:
            path.append(i)

    ans = []
    while path: # path is always used for storing no in-degree nodes
        currentCourse = path.popleft()
        ans.append(currentCourse)
        neighborList = graph[currentCourse]
        for neighbor in neighborList:
            in_degree[neighbor] -= 1 # if ever looped, disconnect the edge
            if in_degree[neighbor] == 0:
                path.append(neighbor)
    if len(ans) == numCourses:
        ans
    else:# not be able to go through all nodes
        ans=[]
    return ans

def snakesAndLadders(self, board: List[List[int]]) -> int: # Skip
    print('909. Snakes and Ladders')
    return 0

def letterCombinations(digits: str) -> List[str]:# others' solution matrix multiplication
    print('17. Letter Combinations of a Phone Number(Medium)')
    result = ['']
    if not digits:
        return []
    num_alpha_dict = collections.defaultdict(list)
    num_alpha_dict['1'] = ''
    num_alpha_dict['2'] = 'abc'
    num_alpha_dict['3'] = 'def'
    num_alpha_dict['4'] = 'ghi'
    num_alpha_dict['5'] = 'jkl'
    num_alpha_dict['6'] = 'mno'
    num_alpha_dict['7'] = 'pqrs'
    num_alpha_dict['8'] = 'tuv'
    num_alpha_dict['9'] = 'wxyz'
    num_alpha_dict['0'] = ''
    digit_q = collections.deque(list(digits))
    for i in range(len(digit_q)):
        current_round = []
        digit = digit_q.popleft()
        for j in num_alpha_dict[digit]:
            for r in result:
                current_round.append(r+j) # old * new coming
        result=current_round # overwrite
    return result

def combine(n: int, k: int) -> List[List[int]]:# my solution is wrong
    print('77. Combinations(Medium)')
    result = []
    need_num_list =[]
    if k==1:
        for i in range(n):
            tmp =[]
            tmp.append(i+1)
            result.append(tmp)
        return result

    for i in range(n):
        need_num_list.append(i + 1)
    length_need_num = len(need_num_list)

    if length_need_num==1 and k==1:
        result.append(need_num_list)
        return result
    elif length_need_num==1 and k<length_need_num:
        return result
    elif length_need_num == 1 and k > length_need_num:
        return result

    for i in range(length_need_num-1):
        for m in range(k-1):
          for j in range(i+1,length_need_num,1):
             tmp = []
             tmp.append(need_num_list[i])
             tmp.append(need_num_list[j])
             result.append(tmp)
    return result

def combine2(n: int, k: int) -> List[List[int]]:# Others' solution
    print('77. Combinations(Medium)')
    result = []

    def backtrack(start,combine):
        if len(combine) == k:
            result.append(combine.copy())
            return

        for i in range(start,n+1):
            combine.append(i)
            backtrack(i+1,combine)
            combine.pop()
    backtrack(1,[])

    return result

def permute(nums: List[int]) -> List[List[int]]:# others' solution, I prefer this solution
    print('46. Permutations(Medium)')
    result = []
    used = [False] * len(nums)

    def dfs(path : List[int]):
        if len(path) == len(nums):
            result.append(path.copy())
            return

        for i, num in enumerate(nums):
            # tmp=[]
            # tmp.append(nums[i])
            # for j in range(len(nums)):
            #     if j==i:
            #         continue
            #     tmp.append(nums[j])
            #     result.append(tmp.copy())
            #     tmp.pop()
            #     print("123")
            if used[i]:
               continue
            used[i] = True
            path.append(num)
            dfs(path)
            path.pop()
            used[i]=False

    dfs([])
    return result

def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    print('39. Combination Sum(Medium)')
    result = []

    def dfs(start, goal, path):
        if goal < 0:
            return
        if goal ==0:
            result.append(path.copy())
            return

        for i in range(start,len(candidates)):
            path.append(candidates[i])
            dfs(i, goal-candidates[i],path)
            path.pop()

    dfs(0,target,[])
    return result

def generateParenthesis(n: int) -> List[str]:
    print('22. Generate Parentheses(Medium)')
    result = []
    def dfs(l, r, s):
        if l == 0 and r == 0:
            result.append(s)
            return
        if l > 0:
            dfs(l-1, r, s+'(')
            print('123')
        if l < r:
            dfs(l, r-1, s + ')')
            print('456')
        print('789')

    dfs(n, n, '')
    return result

def exist(board: List[List[str]], word: str) -> bool:
    find_word_list = []
    status = False

    for i in range(len(board)):
        for j in range(len(board[0])):
          if board[i][j] == word:
             return True

    def dfs(m, row, col, search_word):

      if m == len(word):
          find_word_list.append(search_word)
          return
      if board[row][col] == word[m]:
          search_word = search_word+board[row][col]
          board[row][col]='1'
          if row-1 >= 0:
              dfs(m+1, row-1, col, search_word)
          if row+1 < len(board):
              dfs(m+1, row+1, col, search_word)
          if col-1 >= 0:
              dfs(m+1, row, col-1, search_word)
          if col+1 < len(board[0]):
              dfs(m+1, row, col+1, search_word)
      # if for i in range(row, len(board)):
      #         for j in range(col+1, len(board[0])):
      #             dfs(m, i, j, search_word)


    for i in range(len(board)):
        for j in range(len(board[0])):
          if board[i][j] == word:
             return True
          else:
             if board[i][j] == word[0]:
                 dfs(0, i, j, '')

    for find_word in find_word_list:
        if find_word == word:
            status = True
            break
        else:
            continue
    print('123')
    return status

if __name__ == '__main__':

    '''
     Start from 2024-04-20  
     Binary Tree BFS
    '''

    # # 199. Binary Tree Right Side View(Medium)
    # treeNodeLeft = TreeNode(2,right=TreeNode(5))
    # treeNodeRight = TreeNode(3, right=TreeNode(4))
    # treeNodeRight = TreeNode(3)
    # treeNode = TreeNode(1,treeNodeLeft,treeNodeRight)
    # # rightSideView(root = treeNode)
    # rightSideView(root=[])

    # # 637. Average of Levels in Binary Tree(Easy)
    # # treeNodeLeft = TreeNode(9)
    # # treeNodeRight = TreeNode(20, left=TreeNode(15), right=TreeNode(7))
    # treeNodeLeft = TreeNode(9, left=TreeNode(15), right=TreeNode(7))
    # treeNodeRight = TreeNode(20)
    # treeNode = TreeNode(3,treeNodeLeft,treeNodeRight)
    # averageOfLevels(treeNode)

    # # 102. Binary Tree Level Order Traversal(Medium)
    # treeNodeLeft = TreeNode(9)
    # treeNodeRight = TreeNode(20, left=TreeNode(15), right=TreeNode(7))
    # # treeNodeLeft = TreeNode(9, left=TreeNode(15), right=TreeNode(7))
    # # treeNodeRight = TreeNode(20)
    # treeNode = TreeNode(3,treeNodeLeft,treeNodeRight)
    # levelOrder(treeNode)

    # # 103. Binary Tree Zigzag Level Order Traversal(Medium)
    # treeNodeLeft = TreeNode(9)
    # treeNodeRight = TreeNode(20, left=TreeNode(15), right=TreeNode(7))
    # treeNode = TreeNode(3,treeNodeLeft,treeNodeRight)
    # zigzagLevelOrder(treeNode)


    # '''
    #  Binary Search Tree
    #  don't forget the feature of Binary Search Tree is that
    #  for any root nodes,all left nodes should be less than roo node,
    #  and each of right nodes should larger than roots, each the left nodes should less than roots，
    #  Is kind of sorted tree
    # '''
    # # 530. Minimum Absolute Difference in BST(Easy)
    # # treeNodeLeft = TreeNode(2, left=TreeNode(1), right=TreeNode(3))
    # # treeNodeRight = TreeNode(6)
    # # treeNode = TreeNode(4,treeNodeLeft,treeNodeRight)
    # '''     4
    #    2        6
    #  1   3
    # '''
    # # treeNodeLeft = TreeNode(0)
    # # treeNodeRight = TreeNode(48, left = TreeNode(12), right=TreeNode(49))
    # # treeNode = TreeNode(1, treeNodeLeft, treeNodeRight)
    # '''   1
    #    0      48
    #         12   49
    # '''
    # # [1,null,5,3]
    # # treeNodeRight = TreeNode(5, left = TreeNode(3))
    # # treeNode = TreeNode(1, right=treeNodeRight)
    # '''   1
    #    0      5
    #         3
    # '''
    # # [543,384,652,null,445,null,699]
    # treeNodeLeft = TreeNode(384, right=TreeNode(445))
    # treeNodeRight = TreeNode(652, right=TreeNode(699))
    # treeNode = TreeNode(543, treeNodeLeft, treeNodeRight)
    # '''      543
    #    384         652
    #       445         699
    # '''
    # getMinimumDifferenceOthers(treeNode)

    # # 230. Kth Smallest Element in a BST(Medium)
    # # [3,1,4,null,2]
    # treeNodeLeft = TreeNode(1, right=TreeNode(2))
    # treeNodeRight = TreeNode(4)
    # treeNode = TreeNode(3, treeNodeLeft, treeNodeRight)
    # '''      3
    #    1          4
    #      2
    # '''
    # k = 3
    # kthSmallest(treeNode, k)

    # # 98. Validate Binary Search Tree(Medium)
    # treeNodeLeft = TreeNode(1)
    # treeNodeRight = TreeNode(3)
    # treeNode = TreeNode(2, treeNodeLeft, treeNodeRight)
    # '''      2
    #    1          3
    # '''
    # treeNode = TreeNode(0)
    # '''      0
    # '''
    # isValidBST(treeNode)

    # Graph General
    # 200. Number of Islands(Medium)
    # grid = [
    #     ['1', '1', '1', '1', '0'],
    #     ['1', '1', '0', '1', '0'],
    #     ['1', '1', '0', '0', '0'],
    #     ['0', '0', '0', '0', '0']
    # ]
    #
    # '''
    #   grid = [
    #     ['1'/,'1'&,'1'#,'1'@, '0'],
    #     ['1'&,'1'#,'0'@, '1', '0'],
    #     ['1'#,'1'@, '0', '0', '0'],
    #     ['0'@, '0', '0', '0', '0']
    # ]
    # '''
    # numIslands(grid)

    # # 130. Surrounded Regions(Medium)
    # board = [
    #     ['X','X','X','X'],
    #     ['X','O','O','X'],
    #     ['X','X','O','X'],
    #     ['X','O','X','X']
    # ]
    # solve(board)

    # # 133. Clone Graph(Medium)
    # # adjList = [[2, 4], [1, 3], [2, 4], [1, 3]]
    # node1 = Node(1)
    # node2 = Node(2)
    # node3 = Node(3)
    # node4 = Node(4)
    # node1.neighbors = [node2, node4]
    # node2.neighbors = [node1, node3]
    # node3.neighbors = [node2, node4]
    # node4.neighbors = [node3, node1]
    # cloneGraph(node1)

    # # 399. Evaluate Division(Medium)
    # equations = [["a", "b"], ["b", "c"]]
    # values = [2.0, 3.0]
    # queries = [["a", "c"], ["b", "a"], ["a", "e"],["a", "a"], ["x", "x"]]
    # # [6.00000,0.50000,-1.00000,1.00000,-1.00000]
    #
    # # equations = [["a", "b"], ["b", "c"], ["bc", "cd"]]
    # # values = [1.5, 2.5, 5.0]
    # # queries = [["a", "c"], ["c", "b"],["bc", "cd"], ["cd", "bc"]]
    # # # [3.75000,0.40000,5.00000,0.20000]
    #
    # # equations = [["a", "b"]]
    # # values = [0.5]
    # # queries = [["a", "b"], ["b", "a"], ["a", "c"], ["x", "y"]]
    # # # [0.50000,2.00000,-1.00000,-1.00000]
    # calcEquation2(equations, values, queries)

    # 207. Course Schedule(Medium)
    # numCourses = 2
    # prerequisites = [[1, 0], [0, 1]]
    #
    # numCourses = 2
    # prerequisites = [[0, 1]]

    # numCourses = 4
    # prerequisites = [[1, 0], [2, 1], [3, 1], [3,2]]
    # canFinish2(numCourses,prerequisites)


    # # 210. Course Schedule II(Medium)
    # numCourses = 4
    # prerequisites = [[1, 0], [2, 0], [3, 1], [3,2]]
    #
    # # numCourses = 2
    # # prerequisites = [[1, 0]]
    #
    # # numCourses = 1
    # # prerequisites = []
    #
    # # numCourses = 2
    # # prerequisites = [[0,1]]
    #
    # # numCourses = 2
    # # prerequisites = [[0,1],[1,0]]
    # findOrder2(numCourses, prerequisites)

    # Graph BFS
    # 909. Snakes and Ladders(Medium) skipped

    # # Backtracking
    # # 17. Letter Combinations of a Phone Number(Medium)
    # digits = "23"
    # letterCombinations(digits)

    # # 77. Combinations(Medium)
    # n = 4
    # k = 3
    #
    # # n = 2
    # # k = 1
    # combine2(n,k)
    # print('000')

    # # 46. Permutations(Medium)
    # nums = [1, 2, 3]
    # permute(nums)

    # # 39. Combination Sum(Medium)
    # candidates = [2, 3, 6, 7]
    # target = 7
    # combinationSum(candidates, target)

    # # 22. Generate Parentheses(Medium)
    # n = 3
    # generateParenthesis(n)

    # 79. Word Search(Medium)
    # board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    # word = "ABCCED"

    # board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    # word = "SEE"

    # board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    # word = "ABCB"
    #
    # board = [["a"]]
    # word = "a"

    # board = [["a", "b"], ["c", "d"]]
    # word = "cdba"

    board = [["C", "A", "A"], ["A", "A", "A"], ["B", "C", "D"]]
    word = "AAB"
    exist(board, word)


