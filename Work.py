from PIL import Image, ImageDraw
from math import sqrt
import numpy as np
import random
import pickle
from tqdm import tqdm

class Work :
    def __init__(self) -> None:
        self.block_width = 4
        self.block_height = 4
        self.first_layer_neuron = 6
        self.matrix_update = []
        self.X = []
        self.Y = []
        self.dX = []
        self.blocks_matrix=[]
        self.select_action()
        
    
    def select_action(self, input):
        matrix = input("Enter city name")
        input = input("Enter country code")
        return matrix
    
    def action_1(self):
        app_id = 'Replace with yours'
        details = get_details()
        city_name = details.get('city')
        country_code = details.get('country_code')

        data = requests.post(api_call)
        humidity = []
        date = []
        temp = []
        desc = []
        pressure = []
        sea_level = []

        data = data.json()
        print("\t" + "Details about the forcast of the city " + city_name + " of next 5 days " + "\n\n")
        for lists in data['list']:
            date.append(lists['dt_txt'])
            humidity.append(lists['main']['humidity'])
            temp.append(lists['main']['temp'])
            pressure.append(lists['main']['pressure'])
            sea_level.append(lists['main']['sea_level'])
            desc.append(lists['weather'][0]['description'])

        print('{:^25}{:^20}{:^20}{:^20}{:^20}{:^20}'.format("Date", "Description", "Temprature", "Humidity", "Pressure", "Sea_level\n"))
        for i in range(len(humidity)):
            print('{:^25}{:^20}{:^20}{:^20}{:^20}{:^20}'.format(str(date[i]), desc[i], str(temp[i])+" C\N{DEGREE SIGN}", str(humidity[i])+" %",
                str(pressure[i])+" hPa", str(sea_level[i])+" hPa"))

        with open("test.txt", "a+") as file:
            date = datetime.datetime.now().strftime("%H:%M:%S")
            file.write('Logged data at: ' + str(date) + '\n')

        if cronjob:
            time.sleep(5)
            get_data()

    def action_2(self):
        for t in range(l):
        action = agent.act(state)


        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0:  
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

    
    def action_3(self):
         total_profit = 0
        agent.inventory = []
        for t in range(l):
            action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0: 
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
        
    def action_4(self):
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Loading files from {folder_path}...")
            for file in os.listdir(folder_path):
                try:
                    image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    images.append(image)
                    labels.append(int(folder))
                except Exception as e:
                    print(f"There is a problem with file: {file}")
                    print(str(e))

        return images, labels
    
    def read_Y():
       if encrypt:
    			filenameWithExt = os.path.basename(path) + '.aes'
			vaultpath = self.hid_dir + filenameWithExt
			pyAesCrypt.encryptFile(path, vaultpath, self.key.decode(), self.buffer_size)
		else:
			shutil.copy(path, self.hid_dir)
        return matrix
        
    def compress_image(compress_img, width, height):
        if filenameWithExt.endswith('.aes'):
    		filename = filenameWithExt[:-4]
		pyAesCrypt.decryptFile(vaultpath, filename, self.key.decode(), self.buffer_size)
        
    def read_weight(self, number):
    if len(sys.argv) not in [2, 3]:

    images, labels = load_data(sys.argv[1])
    
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    model = get_model()

    model.fit(x_train, y_train, epochs=EPOCHS)
    model.evaluate(x_test,  y_test, verbose=2)
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
    
    
    def trans_image_to_matrix(self):
        matrix = self.masterpwd.encode()
	    
	    kdf = PBKDF2HMAC(algorithm = hashes.SHA256(),
	                     length = length,
	                     salt = salt,
	                     iterations = 100000,
	                     backend = default_backend())
	    
	    self.blocks = new_matrix(kdf.derive(matrix))

        return result_matrix
    
    def update(self, matrix, height, width):
        if os.path.exists(path):
    		masterpwd = getpass("Enter your Master Password : ")
		vault = secret_vault(masterpwd)
		vault.generate_key()
		fernet = Fernet(vault.key)
        return matrix
    
    def line_block(self, matrix,width,height):
        block = []
         for face_loc, id in zip(face_locations, ids):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            name=""
            if id=="Unknown":
                color=(0,0,200)
                name="Unknown"
            else:
                color=(0,200,0)
                query="UPDATE USER SET location=?, time= ?  where id="+id
                data=('Hostel O', time.time())
                with con:
                    con.execute(query, data)
                with con:
                    data = con.execute("SELECT name FROM USER WHERE id= "+ id)
                    for row in data:
                        name=row[0]
        return block
    
    def create_weight_matrix(self, first_neuron, blocks):
       parser = argparse.ArgumentParser("check_copyright_notice")
        parser.add_argument(
        "--directory", type=str, default=".", help="The path to the repository root."
    )
        return matrix   
    
    def transfom(self, matrix):
        while(True):
        userinput = input("Enter the price: ")
        while type(userinput) != str or userinput.isnumeric() != True:
        userinput = input("Please re-enter a number: ")
        userinput = int(userinput)

    if(userinput != 0):
        Sum = (Sum + userinput)
        total_items = total_items + 1
    else:
       break
        return matrix
    
    def multiple_matrix(self, first_matrix, second_matrix):
       while type(name) != str or name.isalpha() != True:
        name = input("Please re-enter customer's name without any spaces: ") 
    
    def div_matrix(self, first_matrix, second_matrix):
        hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Moning")
    elif hour >= 12 and hour < 18:
        speak("Good afternoon")
    else:
        speak("Good evening")
        return matrix
    
    def draw_image(self, matrix, width, height):
        r.pause_threshold = 0.5
        audio = r.listen(source)
        r.adjust_for_ambient_noise(source)
    try:
        quary = r.recognize_google(audio, language='en-in')

    def image_to_pixels(self, matrix, X, height, width):
        ret, frame1 = cam.read()
        ret, frame2 = cam.read()

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return matrix
    
    def sum_matrix(self, first_matrix, second_matrix):
        matrix = []
        for i_id in range(len(first_matrix)):
            line = []
            for j_id in range(len(first_matrix[0])):
                line.append(first_matrix[i_id][j_id] + second_matrix[i_id][j_id])
            matrix.append(line)
        return matrix

    def devision_matrix(self, alpha, matrix):
         for c in contours:
            if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        winsound.Beep(1000, 500)
        return matrix

    def correct_weight_matrix(self, weight_matrix_1, weight_matrix_2, Y, dX, X=[], layer=1, alphaX = 0.005, alphaY=0.005):
        morse_code=""
        plain_text=""
        for letters in text:
            if letters != " ":
                morse_code +=letters
                space_found=0
            else:
                space_found += 1
                if space_found==2:
                    plain_text += " "
                else:
                    plain_text= plain_text + key_list[val_list.index(morse_code)]
                    morse_code= ""    

    def to_second_step(self, x):
        return  x ** 2

    def sqrt_error(self, dX):
        sqrt = util.get_edited_thumbnail_img(Platforms.youtube, video_id)
        return sqrt
    
    def save_weight(self, weight_matrix, number):
         train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the validation curve
        pl.figure(figsize=(7, 5))
        pl.title('Decision Tree Regressor Complexity Performance')
        pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')
        pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')
        pl.fill_between(max_depth, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        pl.fill_between(max_depth, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')