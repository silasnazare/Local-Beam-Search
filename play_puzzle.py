import argparse
import fileinput
from eight_puzzle import Puzzle

if __name__=="__main__":
	# Initializar o quebra-cabeça
	puzzle = Puzzle()

	# Crie um analisador de linha de comando e adicione argumentos apropriados
	parser = argparse.ArgumentParser(description="Interaja com o Oito Puzzle")
	parser.add_argument('-setState', help='Insira um estado para o quebra-cabeça no formato "b12 345 678"', metavar = '')
	parser.add_argument('-randomizeState', help='Insira um número inteiro de etapas aleatórias para retroceder da meta', type=int, metavar = '')
	parser.add_argument('-printState', action="store_true", help='Exibir o estado atual do quebra-cabeça')
	parser.add_argument('-move', help='Mova o bloco em branco na direção especificada', metavar = '')
	parser.add_argument("-solveAStar", help='Resolva o quebra-cabeça oito usando a heurística especificada', metavar = '')
	parser.add_argument("-solveBeam", help='Resolva o quebra-cabeça oito usando a busca de feixe local com o número especificado de estado',type=int, metavar = '')
	parser.add_argument("-maxNodes", help='Especifique um número máximo de nós para explorar durante a pesquisa', default=None, type=int, metavar = '')
	parser.add_argument("-prettyPrintState", help='Exibir o estado atual de uma maneira esteticamente agradável', action="store_true")
	parser.add_argument("-prettyPrintSolution", help='Exiba o caminho da solução de maneira esteticamente agradável', action="store_true")
	parser.add_argument("-readCommands", help="Leia e execute uma série de comandos de um arquivo de texto onde os comandos são especificados sem traços")

	# Args são todos os argumentos fornecidos
	args = parser.parse_args()

	# Série de instruções if para lidar com argumentos
	if args.setState:
		puzzle.set_state(args.setState)

	if args.randomizeState:
		puzzle.randomize_state(args.randomizeState)

	if args.printState:
		print("Estado atual do quebra-cabeça:")
		puzzle.print_state()

	if args.move:
		puzzle.state = puzzle.move(puzzle.state, args.move)	

	if args.solveAStar:
		if args.maxNodes:
			puzzle.a_star(heuristic = args.solveAStar, max_nodes = args.maxNodes)
		else:
			puzzle.a_star(heuristic = args.solveAStar)

	if args.solveBeam:
		if args.maxNodes:
			puzzle.local_beam(k = args.solveBeam, max_nodes = args.maxNodes)
		else:
			puzzle.local_beam(k = args.solveBeam)

	

	if args.prettyPrintState:
		puzzle.pretty_print_state(puzzle.state)

	if args.prettyPrintSolution:
		puzzle.success(puzzle.expanded_nodes, puzzle.num_nodes_generated, print_solution=False) 
		puzzle.pretty_print_solution(puzzle.solution_path)

	# Se um arquivo de texto for fornecido, é necessário iterar pelos comandos no arquivo
	if args.readCommands:

		# Abra o arquivo de texto e leia linha por linha
		with open(args.readCommands, "r") as f:
			for line in f:
				# Divida os comandos em uma lista em espaços
				arguments = line.split(" ")

				# Retire o caractere de nova linha do último argumento
				arguments[-1] = arguments[-1].strip()

				# Iterar pelos argumentos
				for position, argument in enumerate(arguments):
					if argument == "setState":
						puzzle.set_state(" ".join(arguments[position + 1: position + 4]).strip('\"'))

					elif argument == "randomizeState":
						puzzle.randomize_state(int(arguments[position + 1]))

					elif argument == "move":
						puzzle.state = puzzle.move(puzzle.state, arguments[position + 1].strip('\"'))

					elif argument == "solveAStar":
						heuristic = arguments[position + 1]
						try: 
							max_nodes_index = arguments.index("maxNodes")
							max_nodes = int(arguments[max_nodes_index])
							puzzle.a_star(heuristic = heuristic, max_nodes = max_nodes)
						except:
							puzzle.a_star(heuristic = heuristic)

					elif argument == "solveBeam":
						k = int(arguments[position + 1])
						try: 
							max_nodes_index = arguments.index("maxNodes")
							max_nodes = int(arguments[max_nodes_index])
							puzzle.local_beam(k = k, max_nodes = max_nodes)
						except:
							puzzle.local_beam(k = k)

					elif argument == "printState":
						print("Estado atual do quebra-cabeça:")
						puzzle.print_state()

					elif argument == "prettyPrintState":
						puzzle.pretty_print_state(puzzle.state)

					elif argument == "prettyPrintSolution":
						puzzle.success(puzzle.expanded_nodes, puzzle.num_nodes_generated, print_solution=False)
						puzzle.pretty_print_solution(puzzle.solution_path)