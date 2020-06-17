import sys

args = {'gpu' : '0', 'epoch' : 100, 'learning_rate' : 0.01, 'batch_size' : 8, 'load' : None, 'save' : None, 'class' : None}
float_args = ['learning_rate']
str_args = ['gpu', 'load', 'save', 'class']
args_parse = 0 # 1 if args have been parsed

def error_message(info) :
	print('Error :', info)
	quit()

def print_args() :
	if not args_parse :
		get_args()
	else :
		print('\nProgram :', sys.argv[0])
		print('\nArgument :')
		for arg in args :
			print('%-20s    '%arg, args[arg])
		print('')
	
def get_args() :
	global args_parse
	if args_parse :
		return args
	for index in range(1, len(sys.argv), 2) :
		arg = sys.argv[index][2:]
		if sys.argv[index][:2] == '--' and arg in args :
			if arg in float_args :
				args[arg] = float(sys.argv[index + 1])
			elif arg in str_args:
				args[arg] = sys.argv[index + 1]
			else :
				args[arg] = int(sys.argv[index + 1])
		else :
			error_message('Unrecognized argument : ' + sys.argv[index])
	args_parse = 1
	print_args()
	return args

