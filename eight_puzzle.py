import random 
import copy

class Puzzle():
    # Representação do jogo-dos-oito com métodos para resolução

    def __init__(self):
        # Defina o estado inicial como o estado do objetivo
        random.seed(42)
        self.goal_state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        
    def goal_check(self, state):
        # Verifique se o estado fornecido é o estado do objetivo
        return state == self.goal_state
    
    def set_goal_state(self):
        # Defina o estado do quebra-cabeça para o estado do objetivo
        self.state = copy.deepcopy(self.goal_state)
    
    def get_available_actions(self, state):
        # Retorna a lista de movimentos disponíveis e a localização do espaço em branco, dependendo do estado do quebra-cabeça

        # Encontre a peça em branco
        for row_index, row in enumerate(state):
            for col_index, element in enumerate(row):
                if element == 0:
                    blank_row = row_index
                    blank_column = col_index
        
        # Definir movimentos disponíveis para uma lista vazia
        available_actions = []
        
        # Encontre movimentos disponíveis
        if blank_row == 0:
            available_actions.append("down")
        elif blank_row == 1:
            available_actions.extend(("up", "down"))
        elif blank_row == 2:
            available_actions.append("up")
            
        if blank_column == 0:
            available_actions.append("right")
        elif blank_column == 1:
            available_actions.extend(("left", "right"))
        elif blank_column == 2:
            available_actions.append("left")
            
        # Embaralhe aleatoriamente as ações para remover o viés de ordenação
        random.shuffle(available_actions)

        return available_actions, blank_row, blank_column
    
    def set_state(self, state_string):
        # Defina o estado do quebra-cabeça para uma string no formato "b12 345 678"

        # Verifique o tamanho correto da String
        if len(state_string) != 11:
            print("String Length is not correct!")

        # Acompanhe os elementos que foram adicionados ao quadro
        added_elements = []

        # Enumerar todas as posições na String
        for row_index, row in enumerate(state_string.split(" ")): 
            for col_index, element in enumerate(row):
                # Verifique se não há caracteres inválidos na String
                if element not in ['b', '1', '2', '3', '4', '5', '6', '7', '8']:
                    print("Invalid character in state:", element)
                    break
                else:
                    if element == "b":
                        # Verifique se o espaço em branco foi adicionado duas vezes
                        if element in added_elements:
                            print("The blank was added twice")
                            break

                        # Defina a peça em branco para um 0 no quebra-cabeça
                        else:
                            self.state[row_index][col_index] = 0
                            added_elements.append("b")
                    else:
                        # Verifique se o bloco já foi adicionado ao tabuleiro
                        if int(element) in added_elements:
                            print("Tile {} has been added twice".format(element))
                            break

                        else:
                            # Coloque o bloco correto no tabuleiro
                            self.state[row_index][col_index] = int(element)
                            added_elements.append(int(element))
      
    def randomize_state(self, n):
        # Faça uma série aleatória de movimentos para trás do estado objetivo
        # Define o estado atual do quebra-cabeça para um estado que é garantido como solucionável

        current_state = (self.goal_state)
        
        # Iterar através do número de movimentos
        for i in range(n):
            available_actions, _, _ = self.get_available_actions(current_state)
            random_move = random.choice(available_actions)
            current_state = self.move(current_state, random_move)
          
        # Defina o estado do quebra-cabeça para o estado aleatório
        self.state = current_state
            
    def move(self, state, action):
        # Mova o espaço em branco na direção especificada
        # Retorna o novo estado resultante da movimentação
        available_actions, blank_row, blank_column = self.get_available_actions(state)

        new_state = copy.deepcopy(state)

        # Verifique se a ação é permitida, dado o estado da placa
        if action not in available_actions:
            print("Move not allowed\nAllowed moves:", available_actions)
            return False

        # Execute o movimento como uma série de instruções if
        else:
            if action == "down":
                tile_to_move = state[blank_row + 1][blank_column]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row + 1][blank_column] = 0
            elif action == "up":
                tile_to_move = state[blank_row - 1][blank_column]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row - 1][blank_column] = 0
            elif action == "right":
                tile_to_move = state[blank_row][blank_column + 1]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row][blank_column + 1] = 0
            elif action == "left":
                tile_to_move = state[blank_row][blank_column - 1]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row][blank_column -1] = 0
                
        return new_state


    def print_state(self):
        # Exibe o estado da placa no formato "b12 345 678"
        str_state = []

        # Iterar por todos os ladrilhos
        for row in self.state:
            for element in row:
                if element == 0:
                    str_state.append("b")
                else:
                    str_state.append(str(element))
        
        # Imprima o resultado
        print("".join(str_state[0:3]), "".join(str_state[3:6]), "".join(str_state[6:9]))

    def pretty_print_state(self, state):
        print("\nCurrent State")
        for row in (state):
            print("-" * 13)
            print("| {} | {} | {} |".format(*row))

    def pretty_print_solution(self, solution_path):
        # Exiba o caminho da solução de maneira esteticamente agradável
        try:
            # O caminho da solução está na ordem inversa
            for depth, state in enumerate(solution_path[::-1]):
                if depth == 0:
                    print("\nStarting State")

                elif depth == (len(solution_path) - 2):
                    print("\nGOAL!!!!!!!!!")
                    for row_num, row in enumerate(state[0]):
                        print("-" * 13)
                        print("| {} | {} | {} |".format(*row))

                    print("\n")
                    break
                else:
                    print("\nDepth:", depth)
                for row_num, row in enumerate(state[0]):
                    print("-" * 13)
                    print("| {} | {} | {} |".format(*row))
        except:
            print("No Solution Found")
            
        
    def calculate_h1_heuristic(self, state):
        # Ccalcular e retornar a heurística h1 para um determinado estado
        # A heurística h1 é o número de peças fora do lugar de sua posição de objetivo

        # Achatar as listas para comparação
        state_flat_list = sum(state, [])
        goal_flat_list = sum(self.goal_state, [])
        heuristic = 0

        # Iterar pelas listas e comparar elementos
        for i, j in zip(state_flat_list, goal_flat_list):
            if i != j:
                heuristic += 1

        
        return heuristic

    def calculate_h2_heuristic(self, state):
        # Calcula e retorna a heurística h2 para um determinado estado
        # A huerística h2 para o jogo-dos-oito é definida como a soma das distâncias de Manhattan de todas as peças
        # A distância de Manhattan é a soma do valor absoluto da diferença x e y da posição atual do ladrilho em relação à posição do estado do objetivo

        state_dict = {}
        goal_dict = {}
        heuristic = 0
        
        # Crie dicionários do estado atual e do estado da meta
        for row_index, row in enumerate(state):
            for col_index, element in enumerate(row):
                state_dict[element] = (row_index, col_index)
        
        for row_index, row in enumerate(self.goal_state):
            for col_index, element in enumerate(row):
                goal_dict[element] = (row_index, col_index)
                
        for tile, position in state_dict.items():
            # Não conte a distância do espaço em branco
            if tile == 0:
                pass
            else:
                # Calcular heurística como a distância de Manhattan
                goal_position = goal_dict[tile]
                heuristic += (abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1]))

        return heuristic

    def calculate_total_cost(self, node_depth, state, heuristic):
        # Retorna o custo total de um estado dado sua profundidade e a heurística
        # O custo total em uma estrela é o custo do caminho mais a heurística. O custo do caminho neste caso é a profundidade ou o número de movimentos do estado inicial para o estado atual porque todos os movimentos têm o mesmo custo

        if heuristic == "h2":
            return node_depth + self.calculate_h2_heuristic(state)
        elif heuristic == "h1":
            return node_depth + self.calculate_h1_heuristic(state)
    
    def a_star(self, heuristic="h2", max_nodes=10000, print_solution=True):
        # Executa uma pesquisa de uma estrela
        # Imprime a lista de movimentos de solução e o tamanho da solução

        # Precisa de um dicionário para a fronteira e para os nós expandidos
        frontier_nodes = {}
        expanded_nodes = {}
        
        self.starting_state = copy.deepcopy(self.state)
        current_state = copy.deepcopy(self.state)
        # O índice de nós é usado para indexar os dicionários e acompanhar o número de nós expandidos
        node_index = 0

        # Defina o primeiro elemento em ambos os dicionários para o estado inicial
        # Este é o único nó que estará em ambos os dicionários
        expanded_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.calculate_total_cost(0, current_state, heuristic), "depth": 0}
        
        frontier_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.calculate_total_cost(0, current_state, heuristic), "depth": 0}
        

        failure = False

        # O método mantém o controle de todos os nós na fronteira e é a fila de prioridade. Cada elemento na lista é uma tupla que consiste no índice do nó e no custo total do nó. Isso será classificado pelo custo total e servirá como fila de prioridade.
        all_frontier_nodes = [(0, frontier_nodes[0]["total_cost"])]

        # Parar quando o máximo de nós for considerado
        while not failure:

            # Obtenha a profundidade de estado atual para uso no cálculo de custo total
            current_depth = 0
            for node_num, node in expanded_nodes.items():
                if node["state"] == current_state:
                    current_depth = node["depth"]

            # Encontre as ações disponíveis correspondentes ao estado atual
            available_actions, _, _ = self.get_available_actions(current_state)

            # Iterar por meio de ações possíveis
            for action in available_actions:
                repeat = False

                # Se o máximo de nós for atingido, saia do loop
                if node_index >= max_nodes:
                    failure = True
                    print("No Solution Found in first {} nodes generated".format(max_nodes))
                    self.num_nodes_generated = max_nodes
                    break

                # Encontre o novo estado correspondente à ação e calcule o custo total
                new_state = self.move(current_state, action)
                new_state_parent = copy.deepcopy(current_state)

                # Verifique se o novo estado já foi expandido
                for expanded_node in expanded_nodes.values():
                    if expanded_node["state"] == new_state:
                        if expanded_node["parent"] == new_state_parent:
                            repeat = True

                # Verifique se o novo estado e o pai estão na fronteira
                # O mesmo estado pode ser adicionado duas vezes à fronteira se o estado pai for diferente
                for frontier_node in frontier_nodes.values():
                    if frontier_node["state"] == new_state:
                        if frontier_node["parent"] == new_state_parent:
                            repeat = True

                # Se o novo estado já foi expandido ou está na fronteira, continue com a próxima ação
                if repeat:
                    continue

                else:
                    # Cada ação representa outro nó gerado
                    node_index += 1
                    depth = current_depth + 1

                    # O custo total é o comprimento do caminho (número de etapas a partir do estado inicial) + heurística
                    new_state_cost = self.calculate_total_cost(depth, new_state, heuristic)

                    # Adicione o índice do nó e o custo total à lista
                    all_frontier_nodes.append((node_index, new_state_cost))

                    # Adicione o nó à fronteira
                    frontier_nodes[node_index] = {"state": new_state, "parent": new_state_parent, "action": action, "total_cost": new_state_cost, "depth": current_depth + 1}

            # Classifique todos os nós na fronteira pelo custo total
            all_frontier_nodes = sorted(all_frontier_nodes, key=lambda x: x[1])

            # Se o número de nós gerados não exceder o máximo de nós, encontre o melhor nó e defina o estado atual para esse estado
            if not failure:
                # O melhor nó estará na frente da fila
                # Após selecionar o nó para expansão, remova-o da fila
                best_node = all_frontier_nodes.pop(0)
                best_node_index = best_node[0]
                best_node_state = frontier_nodes[best_node_index]["state"]
                current_state = best_node_state

                # Mova o nó da fronteira para os nós expandidos
                expanded_nodes[best_node_index] = (frontier_nodes.pop(best_node_index))
                
                # Verifique se o estado atual é o estado da meta
                if self.goal_check(best_node_state):
                    # Crie atributos para os nós expandidos e os nós de fronteira
                    self.expanded_nodes = expanded_nodes
                    self.frontier_nodes = frontier_nodes
                    self.num_nodes_generated = node_index + 1

                    # Exibir o caminho da solução
                    self.success(expanded_nodes, node_index, print_solution)
                    break 
                    
    def local_beam(self, k=1, max_nodes = 10000, print_solution=True):
        # Executa a busca de feixe local para resolver o quebra-cabeça oito
        # k é o número de estados sucessores a serem considerados em cada iteração
        # A função de avaliação é h1 + h2, a cada iteração, o próximo conjunto de nós serão os k nós com a pontuação mais baixa

        self.starting_state = copy.deepcopy(self.state)
        starting_state = copy.deepcopy(self.state)
        # Verifique se o estado atual já é o objetivo
        if starting_state == self.goal_state:
            self.success(node_dict={}, num_nodes_generated=0)

        # Crie um dicionário de referência de todos os estados gerados
        all_nodes= {}

        # Índice para todos os dicionários de nós
        node_index = 0

        all_nodes[node_index] = {"state": starting_state, "parent": "root", "action": "start"}

        # Pontuação para o estado inicial
        starting_score = self.calculate_h1_heuristic(starting_state) + self.calculate_h2_heuristic(starting_state)

        # Nós disponíveis são todos os estados possíveis que podem ser acessados ​​a partir do estado atual armazenado como uma tupla (índice, pontuação)
        available_nodes = [(node_index, starting_score)]
                
        failure = False
        success = False

        while not failure:

            # Verifique se o número de nós gerados excede o máximo de nós
            if node_index >= max_nodes:
                failure = True
                print("No Solution Found in first {} generated nodes".format(max_nodes))
                break
              
            # Os nós sucessores são todos os nós que podem ser alcançados de todos os estados disponíveis. A cada iteração, isso é redefinido para uma lista vazia
            successor_nodes = []

            # Iterar através de todos os nós possíveis que podem ser visitados
            for node in available_nodes:

                repeat = False

                # Encontre o estado atual
                current_state = all_nodes[node[0]]["state"]

                # Encontre as ações correspondentes ao estado
                available_actions, _, _ = self.get_available_actions(current_state)

                # Iterar através de cada ação que é permitida
                for action in available_actions:
                    # Encontre o estado sucessor para cada ação
                    successor_state = self.move(current_state, action)

                    # Verifique se o estado já foi visto
                    for node_num, node in all_nodes.items():
                        if node["state"] == successor_state:
                            if node["parent"] == current_state:
                                repeat = True

                    # Verifique se o estado é o estado da meta
                    # Se o melhor estado for o objetivo, pare a iteração
                    if successor_state == self.goal_state:	
                        all_nodes[node_index] = {"state": successor_state, 
                                "parent": current_state, "action": action}
                        self.expanded_nodes = all_nodes
                        self.num_nodes_generated = node_index + 1
                        self.success(all_nodes, node_index, print_solution)
                        success = True
                        break

                    if not repeat:
                        node_index += 1
                        # Calcular a pontuação do estado
                        score = (self.calculate_h1_heuristic(successor_state) + self.calculate_h2_heuristic(successor_state))
                        # Adicione o estado à lista de nós
                        all_nodes[node_index] = {"state": successor_state, "parent": current_state, "action": action}
                        # Adicione o estado à successor_nodes list
                        successor_nodes.append((node_index, score))
                    else:
                        continue

                    
            # Os nós disponíveis agora são todos os nós sucessores classificados por pontuação
            available_nodes = sorted(successor_nodes, key=lambda x: x[1])

            # Escolha apenas os k melhores estados sucessores
            if k < len(available_nodes):
                available_nodes = available_nodes[:k]
            if success == True:
            	break  
                
def success(self, node_dict, num_nodes_generated, print_solution=True):
        # Depois que a solução for encontrada, imprime o caminho da solução e o comprimento do caminho da solução
        if len(node_dict) >= 1:

            # Encontre o nó final
            for node_num, node in node_dict.items():
                if node["state"] == self.goal_state:
                    final_node = node_dict[node_num]
                    break

            # Gere o caminho da solução do nó final para o nó inicial
            solution_path = self.generate_solution_path(final_node, node_dict, path=[([[0, 1, 2], [3, 4, 5], [6, 7, 8]], "goal")])
            solution_length = len(solution_path) - 2

        else:
            solution_path = []
            solution_length = 0
        
        self.solution_path = solution_path 

        if print_solution:
            # Exibe o comprimento da solução e o caminho da solução
            print("Solution found!")
            print("Solution Length: ", solution_length)

            # O caminho da solução vai do nó final ao nó inicial. Para exibir a sequência de ações, inverta o caminho da solução
            print("Solution Path", list(map(lambda x: x[1], solution_path[::-1])))
            print("Total nodes generated:", num_nodes_generated)
        
def generate_solution_path(self, node, node_dict, path):
        # Retorna o caminho da solução para exibição do estado final (objetivo) para o estado inicial
        # Se o nó for a raiz, retorne o caminho
        if node["parent"] == "root":
            # Se a raiz for encontrada, adicione o nó e retorne
            path.append((node["state"], node["action"]))
            return path

        else:
            # Se o nó não for a raiz, adicione o estado e a ação ao caminho da solução
            state = node["state"]
            parent_state = node["parent"]
            action = node["action"]
            path.append((state, action))

            # Find the parent of the node and recurse
            for node_num, expanded_node in node_dict.items():
                if expanded_node["state"] == parent_state:
                    return self.generate_solution_path(expanded_node, node_dict, path)