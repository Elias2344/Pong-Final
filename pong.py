import pygame
import numpy as np
import random
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,180,0)
BLUE = (50,200,255)
FILL = BLACK
TEXT = WHITE
pygame.init()
#Here you can specify the structure of the neural network. This includes the input layer and output layer.
#e.g 3 inputs, 5 node hidden layer, 4 outputs would be [3, 5, 4]
#Be sure to update this if you add inputs
layer_structure = [4, 3]
#Initializing the display window
size = (800,600)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("pong")

testCoefs = [np.array([[0.38238344, 0.7515745 , 0.29565119, 0.35490288, 0.97040034],
       [0.33545982, 0.0973694 , 0.41539856, 0.76129553, 0.93089118],
       [0.85154809, 0.0240888 , 0.74555908, 0.34759429, 0.37355357],
       [0.95104127, 0.29077331, 0.21244898, 0.78876218, 0.35243364]]), np.array([[0.25498077, 0.03853811, 0.76089995],
       [0.36535132, 0.60519588, 0.08365677],
       [0.12852428, 0.0156597 , 0.03317768],
       [0.1276382 , 0.13700435, 0.6786845 ],
       [0.71931642, 0.8930938 , 0.24983195]])]


class Paddle:
	
	def __init__(self, x = 400, xspeed = 0, coefs = 0, intercepts = 0):
		self.x = x
		self.xlast = x-xspeed
		self.xspeed = xspeed
		self.alive = True
		self.score = 0
		self.command = 2
		self.winner = False
		if coefs == 0:
			self.coefs = self.generateCoefs(layer_structure)
		else:
			self.coefs = coefs
		if intercepts == 0:
			self.intercepts = self.generateIntercepts(layer_structure)
		else:
			self.intercepts = intercepts
	 
	
	def generateCoefs(self, layer_structure):
		coefs = []
		for i in range(len(layer_structure)-1):
			coefs.append(np.random.rand(layer_structure[i], layer_structure[i+1])*2-1)
		return coefs
	
	def generateIntercepts(self, layer_structure):
		intercepts = []
		for i in range(len(layer_structure)-1):
			intercepts.append(np.random.rand(layer_structure[i+1])*2-1)
		return intercepts
	
	
	def mutateCoefs(self):
		newCoefs = self.coefs.copy()
		for i in range(len(newCoefs)):
			for row in range(len(newCoefs[i])):
				for col in range(len(newCoefs[i][row])):
					newCoefs[i][row][col] = np.random.normal(newCoefs[i][row][col], 1)
		return newCoefs
	
	
	def mutateIntercepts(self):
		newIntercepts = self.intercepts.copy()
		for i in range(len(newIntercepts)):
			for row in range(len(newIntercepts[i])):
				newIntercepts[i][row] = np.random.normal(newIntercepts[i][row], 1)
		return newIntercepts
	
	
	def mutate(self):
		return Paddle(coefs = self.mutateCoefs(), intercepts = self.mutateIntercepts())
		
	
	def reset(self):
		self.x = 400
		self.xlast = 400
		self.xspeed = 0
		self.alive = True
		self.score = 0
	
	
	def update(self):
		self.xlast = self.x
		self.x += self.xspeed
		if self.x < 0:
			self.x = 0
		elif self.x > size[0]-100:
			self.x=size[0]-100
		
		self.xlast = self.x
	
	   
	def draw(self):
		if self.winner == False:
			pygame.draw.rect(screen,BLACK,[self.x,size[1]-20,100,20])
			pygame.draw.rect(screen,RED,[self.x+2,size[1]-18,100-4,20-4])
		else:
			pygame.draw.rect(screen,BLACK,[self.x,size[1]-20,100,20])
			pygame.draw.rect(screen,BLUE,[self.x+2,size[1]-18,100-4,20-4])

class Ball:
	
	def __init__(self, x = 50, y = 50, xspeed = 5, yspeed = 5):
		self.x = x
		self.y = y
		self.xlast = x-xspeed
		self.ylast = y-yspeed
		self.xspeed = xspeed
		self.yspeed = yspeed
		self.alive = True
	
	
	def update(self, paddle):
		self.xlast = self.x
		self.ylast = self.y
		
		self.x += self.xspeed
		self.y += self.yspeed
		
		
		if self.x<0:
			self.x=0
			self.xspeed = self.xspeed * -1
		elif self.x>size[0]-15:
			self.x=size[0]-15
			self.xspeed = self.xspeed * -1
		elif self.y<0:
			self.y=0
			self.yspeed = self.yspeed * -1
		elif self.x>paddle.x and self.x<paddle.x+100 and self.ylast<size[1]-35 and self.y>=size[1]-35:
			self.yspeed = self.yspeed * -1
			paddle.score = paddle.score + 1
		elif self.y>size[1]:
			self.yspeed = self.yspeed * -1
			paddle.alive = False
			paddle.score -= round(abs((paddle.x+50)-self.x)/100,2)
			
		   
	def draw(self):
		pygame.draw.rect(screen,WHITE,[self.x,self.y,15,15])
	

def calculateOutput(input, layer_structure, coefs, intercepts, g="identity"):
	
	layers = [np.transpose(input)]

	previousLayer = np.transpose(input)
	
	reduced_layer_structure = layer_structure[1:]
	
	for k in range(len(reduced_layer_structure)):
		
		currentLayer = np.empty((reduced_layer_structure[k],1))
		
		result = np.matmul(np.transpose(coefs[k]),previousLayer) + np.transpose(np.array([intercepts[k]]))
	
		for i in range(len(currentLayer)):
			if g == "identity":
				currentLayer[i] = result[i]
			elif g == "relu":
				currentLayer[i] = max(0, result[i])
			elif g == "tanh":
				currentLayer[i] = tanh(result[i])
			elif g == "logistic":
				try:
					currentLayer[i] = 1 / (1 + exp(-1*result[i]))
				except OverflowError:
					currentLayer[i] = 0
		
		layers.append(currentLayer)
		previousLayer = currentLayer.copy()
	
	
	return(layers[-1].tolist().index(max(layers[-1].tolist())))	
	

def mutateCoefs(coefs):
	newCoefs = []
	for array in coefs:
		newCoefs.append(np.copy(array))
	for i in range(len(newCoefs)):
		for row in range(len(newCoefs[i])):
			for col in range(len(newCoefs[i][row])):
				newCoefs[i][row][col] = np.random.normal(newCoefs[i][row][col], 1)
	return newCoefs

def mutateIntercepts(intercepts):
	newIntercepts = []
	for array in intercepts:
		newIntercepts.append(np.copy(array))
	for i in range(len(newIntercepts)):
		for row in range(len(newIntercepts[i])):
			newIntercepts[i][row] = np.random.normal(newIntercepts[i][row], 1)
	return newIntercepts 
	

def displayNetwork(layer_sctructure, coefs = testCoefs, command = 0):
	
	
	max_coef = np.max(coefs[0])
	

	height = 300
	width = 300
	
	inputs = ["paddle x", "ball x", "ball y", "ball Xspeed", "ball Yspeed"]
	outputs = ["left", "right", "stop"]
	
	layerCount = len(layer_structure)

	circle_positions = []
	
	#Label inputs
	for i in range(layer_structure[0]):
		font= pygame.font.SysFont('Calibri', 30, False, False)
		text = font.render(inputs[i], True, TEXT)
		screen.blit(text,[0,(i+1)* int(height/(layer_structure[0]+2))])	
	
	#Label outputs
	for i in range(layer_structure[-1]):
		font= pygame.font.SysFont('Calibri', 30, False, False)
		text = font.render(str(outputs[i]), True, TEXT)
		screen.blit(text,[width+50,(i+1)* int(height/(layer_structure[-1]+2))])	
	
	
	xspacing = int( width/(layerCount))
	
	#Determine the location of the neurons for each layer, stores that in a list, and stores those lists in circle_positions
	for i in range(layerCount):
		layer_circle_positions = []
		yspacing = int( height/(layer_structure[i]+2))	
		for j in range(layer_structure[i]):
			layer_circle_positions.append(((i+1)*xspacing, (j+1)*yspacing))
		circle_positions.append(layer_circle_positions)
	
	
	for i in range(len(circle_positions)-1):
		for j, circle_pos in enumerate(circle_positions[i]):
			for k, circle_pos2 in enumerate(circle_positions[i+1]):
				thickness = int(coefs[i][j,k]/max_coef*8)
				
				if thickness > 0:
					pygame.draw.lines(screen, BLUE, False, [circle_pos, circle_pos2], thickness)
				else:
					pygame.draw.lines(screen, RED, False, [circle_pos, circle_pos2], -thickness)
					
	
	for layer in circle_positions:
		for circle_pos in layer:
			pygame.draw.circle(screen, BLACK, circle_pos, 20, 0)
			pygame.draw.circle(screen, GREEN, circle_pos, 16, 0)
			
done = False
score = 0
command = "stop"
clock=pygame.time.Clock()
COUNT = 100
#create sprites
paddles = []
balls = []
for i in range(100):
	paddles.append(Paddle())
	balls.append(Ball())

winner = paddles[-1]
paddles[-1].winner = True

generation = 1
while not done:
	screen.fill(FILL)
	
	
	still_alive = 0
	
	high_score = -9e99
	high_score_index = -1
	

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True
	
	
	for i, paddle in enumerate(paddles):
		
		input = np.array([[paddle.x, balls[i].x, balls[i].y, balls[i].xspeed]])
		paddle.command = calculateOutput(input, layer_structure, paddle.coefs, paddle.intercepts)
		
	
		if paddle.command == 0:
				paddle.xspeed = -5
		elif paddle.command == 1:
				paddle.xspeed = 5
		elif paddle.command == 2:
				paddle.xspeed = 0
	
		if paddle.alive == True:
			paddle.update()  
			balls[i].update(paddle)
			still_alive += 1
		
		if paddle.score > high_score:
			high_score = paddle.score
			high_score_index = i
			winner = paddles[i]
			winner.winner = True
			
	
		if paddle.alive and paddle != winner:
			paddle.draw()
			balls[i].draw()
			paddle.winner = False
	
	
	paddles[high_score_index].draw()
	balls[high_score_index].draw()
		
	
	if still_alive == 0:
		generation += 1
		winner.reset()
		print(high_score_index)
		
		paddles = []
		balls = []
	
		for i in range(COUNT-1):
			paddles.append(Paddle(coefs = mutateCoefs(winner.coefs), intercepts = mutateIntercepts(winner.intercepts)))
			balls.append(Ball())
	
		paddles.append(winner)
		balls.append(Ball())
	

	font= pygame.font.SysFont('Calibri', 50, False, False)
	text = font.render("Score = " + str(high_score), True, TEXT)
	screen.blit(text,[size[0]-300,30])	
	text2 = font.render("Still alive = " + str(still_alive), True, TEXT)
	screen.blit(text2,[size[0]-300, 90])	
	text2 = font.render("Generation = " + str(generation), True, TEXT)
	screen.blit(text2,[size[0]-300, 150])	
	displayNetwork(layer_structure, coefs = winner.coefs)
  
	pygame.display.flip()		 
	clock.tick(60)
	
pygame.quit()