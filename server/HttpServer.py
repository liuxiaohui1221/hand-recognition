import tornado.ioloop
import tornado.web
import json
from server.Main_predict import predict
from model import Model

# load model
model= Model.load_model()

class MainHandler(tornado.web.RequestHandler):
	def get(self):
		pass

	def post(self):
		'''post请求'''
		body = self.request.body
		body_decode = body.decode()
		body_json = json.loads(body_decode)
		# base64
		image_base64 = body_json.get("image")
		# predict
		num=predict(model,image_base64)
		self.write("{\"num\":" + str(num) + "}")


application = tornado.web.Application([(r"/recognize", MainHandler), ])

if __name__ == "__main__":
	application.listen(8088)
	tornado.ioloop.IOLoop.instance().start()