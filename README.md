# SimpleChatBotUsingNN
Giải thích các file:
- chatbot_model.h5 : File lưu mô hình mạng nơ ron nhân tạo sau khi chạy file train_chatbot.py 
		và được dùng trong file chatgui.py để xác định "tag" của câu mà người dùng 
		nhập vào
- chatgui.py : File code thành phẩm để chạy chatbot có giao diện người dùng
- classes.pkl : Lưu các "tag" có trong bộ dữ liệu train
- intents.json : dữ liệu train cũng như dữ liệu để trả lời các câu nói của người dùng nhập
- train_chatbot.py : File code để tạo ra mô hình mạng nơron nhân tạo dùng để nhận biết "tag"
		của một câu
- words.pkl : File lưu các từ vựng có trong bộ dữ liệu train
