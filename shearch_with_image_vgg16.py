import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog
from keras import applications, models
import matplotlib.pyplot as plt
import os
import json

def get_vgg16_model():
    base_model = applications.vgg16.VGG16(weights=None, include_top=True)  
    model = models.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    return model

vgg_model = get_vgg16_model()

def extract_hog_features(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    
    image = cv2.resize(image, (128, 128))  # Resize the image
    hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                        block_norm='L2-Hys', visualize=True)
    return hog_features



def extract_vgg_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # الحجم المطلوب لشبكة VGG16
    image = applications.vgg16.preprocess_input(np.expand_dims(image, axis=0))
    features = vgg_model.predict(image)
    return features.flatten()


# === بناء قاعدة بيانات الصور وحفظها === #
def build_image_database(folder_path, save_path='test.json'):
    database = []
    titels = ["Clean Code: A Handbook of Agile Software Craftsmanship",
"The Pragmatic Programmer: Your Journey to Mastery",
"Code Complete: A Practical Handbook of Software Construction",
"Design Patterns: Elements of Reusable Object-Oriented Software",
"Refactoring, Improving the Design of Existing Code",
"You Don’t Know JS (Yet): Scope & Closures",
"Eloquent JavaScript, A Modern Introduction to Programming",
"Introduction to the Theory of Computation",
"Automate the Boring Stuff with Python",
"Cracking the Coding Interview",
"Salt, Fat, Acid, Heat: Mastering the Elements of Good Cooking",
"The Joy of Cooking",
"How to Cook Everything: Simple Recipes for Great Food",
"Essentials of ClassicItalian Cooking",
"Ottolenghi Simple: A Cookbook",
"The Food Lab: Better Home Cooking Through Science",
"The Noble Qur'an: A New Translation",
"In the Footsteps of the Prophet: Lessons from the Life of Muhammad",
"The Sealed Nectar (Ar-Raheeq Al-Makhtum)",
"The Life of Muhammad",
"Islam, A Short History",
"House of Fear: An Anthology of Haunted House Stories",
" The Blue Elephant (الفيل الأزرق)",
"The Secret of Room 207 (سر الغرفة 207)",
"Smile, You Are Dead (ابتسم فأنت ميت)",
"The Butcher (الجزار)",
"Diamond Dust (تراب الماس)",
"It",
"The Haunting of Hill House",
"Bird Box",
"The Shining",
"House of Leaves",
"The Granada Trilogy (ثلاثية غرناطة)",
"Men in the Sun (رجال في الشمس)",
"The Bamboo Stalk (ساق البامبو)",
"A People's History of the United States",
"Sapiens: A Brief History of Humankind",
"The Silk Roads: A New History of the World",
"The Guns of August",
"Astrophysics for People in a Hurry",
"An Introduction to Modern Astrophysics",
"A Brief History of Time",
"Cosmos",
"The Elegant Universe: Superstrings, Hidden Dimensions, and the Quest for the Ultimate Theory",
"The End of Time: The Next Revolution in Our Understanding of the Universe",
"The Martian",
"Cosmos",
"Packing for Mars: The Curious Science of Life in the Void",
"An Astronaut's Guide to Life on Earth",
"The Right Stuff"]
    Author = ["Robert C.Martin",
" Andrew Hunt and David Thomas",
"Steve McConnell",
" Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides",
"Martin Fowler",
"Kyle Simpson",
"Marijn Haverbeke",
"Michael Sipser",
"Al Sweigart",
"Gayle Laakmann McDowell",
"Samin Nosrat",
" Irma S.Rombauer, Marion Rombauer Becker, and Ethan Becker",
"Mark Bittman",
"Marcella Hazan",
"Yotam Ottolenghi",
"J.Kenji López-Alt",
"Dr.Muhammad Taqi-ud-Din al-Hilali and Dr.Muhammad Muhsin Khan",
"Tariq Ramadan",
"Safi-ur-Rahman al-Mubarakpuri",
"Muhammad Husayn Haykal",
"Karen Armstrong",
"Jonathan Oliver",
"Ahmed Mourad",
"Ahmed Khaled Tawfik",
"Hassan Al-Gendy",
"Hassan Al-Gendy",
"Ahmed Mourad",
"Stephen King",
"Shirley Jackson",
"Josh Malerman",
"Stephen King",
"Mark Z. Danielewski",
"Radwa Ashour",
"Ghassan Kanafani",
"Saud Alsanousi",
"Howard Zinn",
"Yuval Noah Harari",
"Peter Frankopan",
"Barbara Tuchman",
"Neil deGrasse Tyson",
"Bradley W. Carroll & Dale A. Ostlie",
"Stephen Hawking",
"Carl Sagan",
"Brian Greene",
"Julian Barbour",
"Andy Weir",
"Carl Sagan",
"Mary Roach",
"Chris Hadfield",
"Tom Wolfe"]
    Description = [
"This book emphasizes writing readable, maintainable, and efficient code.It provides practical advice on clean coding practices and highlights the principles of simplicity and clarity in software development.",
"A timeless classic that offers practical advice on becoming a better developer.The book covers a range of topics including debugging, effective teamwork, and maintaining code quality, emphasizing adaptability and continuous improvement.",
"An in-depth guide to writing robust software.It covers best practices in software construction, emphasizing the importance of structure, design, and effective coding.",
"A groundbreaking book that introduces 23 design patterns for solving common object-oriented software design challenges.It provides reusable solutions and best practices for creating flexible and reusable designs.",
"This book teaches techniques for improving the structure and design of existing code without altering its functionality.It focuses on making code more efficient and easier to maintain.",
"This book is part of the \"You Don’t Know JS\" series, which dives deeply into JavaScript concepts.It covers scope, closures, and the intricacies of the language, making it ideal for those who want to master JavaScript.",
"A beginner-friendly book that introduces programming and JavaScript, covering essential topics like functions, loops, data structures, and web programming basics",
"This book provides a clear introduction to computational theory, including automata theory, formal languages, and complexity theory.It's a must-read for understanding the theoretical foundations of computer science.",
"Perfect for beginners, this book teaches Python through practical projects like automating tasks, working with spreadsheets, and web scraping.It's an excellent choice for those looking to get started with Python.",
"Focused on preparing for technical interviews, this book provides 189 programming questions and detailed solutions to help you ace coding interviews at top tech companies.",
"This award-winning cookbook breaks cooking into four essential elements: salt, fat, acid, and heat.It teaches you how to master these concepts to create flavorful and balanced dishes.",
"A classic cookbook that has been a staple in American kitchens for decades.It covers a wide variety of recipes, techniques, and tips for home cooking, making it perfect for beginners and experts alike.",
"A comprehensive guide to cooking that covers over 2,000 recipes.It emphasizes simplicity and flexibility, helping cooks of all skill levels create delicious meals.",
"A definitive guide toItalian cuisine, this book combines traditional recipes with detailed explanations of techniques and ingredients.It’s a must-have for lovers ofItalian food.",
"This cookbook features 130 easy and accessible recipes that retain the bold flavors and creativity Ottolenghi is known for.Perfect for busy home cooks who want to prepare exciting meals.",
"A comprehensive exploration of the science of cooking, featuring innovative techniques and over 300 recipes.It’s ideal for those who want to understand the \"why\" behind cooking methods.",
"This is a widely recognized English translation of the Qur'an, with the original Arabic text accompanied by a parenthetical explanation of its meanings based on classical interpretations.The translation aims to make the meanings of the Qur'an more accessible to the English-speaking world.",
"This book offers profound insights into the life and character of Prophet Muhammad (PBUH), focusing on his spiritual, ethical, and personal qualities.Tariq Ramadan explores how his life can be a source of guidance and inspiration for Muslims today.",
"This book is a detailed biography of Prophet Muhammad (PBUH), providing a historical account of his life from birth to death, with emphasis on his leadership, wisdom, and the challenges he faced.It was awarded the first prize in a worldwide competition organized by the Muslim World League.",
"This book provides a detailed, scholarly account of the life of Prophet Muhammad (PBUH), presenting an objective and historically accurate portrayal of his mission, struggles, and achievements in spreading Islam.",
"This concise book by renowned historian Karen Armstrong explores the origins and evolution of Islam.It covers key moments in Islamic history, including the life of Prophet Muhammad (PBUH), the spread of Islam, and the development of the various branches of the faith.",
"This collection features stories from various authors, each exploring the theme of haunted houses and the fears they evoke",
"A psychological thriller about a psychiatrist who returns to work at a mental hospital and discovers a mysterious patient harboring dark secrets.",
"A collection of chilling stories about supernatural events occurring in Room 207 of an old hotel.",
"A gripping tale of a young man who gains supernatural abilities after an accident and becomes the target of dark forces.",
"A dark and twisted narrative about a serial killer leaving cryptic clues, leading investigators into a world of horrifying secrets.",
"A suspenseful story about a pharmacist who uncovers a dark secret in his workplace, dragging him into a web of crime and corruption.",
"A chilling story of seven children who are terrorized by a shape-shifting entity that often takes the form of a clown named Pennywise. The novel delves into themes of childhood fears and the power of friendship.",
"A psychological horror masterpiece about a group of people who investigate paranormal activities in a mysterious mansion, only to find themselves deeply affected by the house's sinister presence.",
"A gripping tale of survival in a post-apocalyptic world where unseen creatures drive people to madness and death upon sight. The protagonist navigates this terrifying world blindfolded to protect herself and her children.",
"The story of a family staying in an isolated hotel during the winter. As the father succumbs to the influence of the hotel's supernatural forces, the family is plunged into terror.",
"A mind-bending horror novel about a family discovering their house is larger on the inside than it is on the outside, leading to terrifying and surreal events.",
"A powerful story set in the final days of Muslim Spain, following the struggles of a Granadan family as they face the fall of their culture and identity.",
"A poignant tale about three Palestinian refugees attempting to smuggle themselves into Kuwait for a better life, highlighting the despair and suffering of the Palestinian diaspora.",
"The story of a young man born to a Kuwaiti father and a Filipina mother, exploring themes of identity, belonging, and cultural conflict. Winner of the 2013 International Prize for Arabic Fiction.",
"This book presents American history from the viewpoint of marginalized groups, challenging traditional narratives and highlighting the struggles of indigenous peoples, African Americans, and other minorities.",
"Harari explores the evolution of Homo sapiens, examining how biology and history have shaped human societies and the world we live in today.",
"This work shifts the focus from Western-centric history to the East, detailing the interconnectedness of civilizations through trade routes and cultural exchanges.",
"Tuchman provides an in-depth analysis of the events leading up to World War I, emphasizing the political miscalculations and diplomatic failures that escalated the conflict.",
"A concise and engaging overview of the universe, covering topics from black holes to quantum mechanics. Tyson presents complex concepts in an accessible manner, making it ideal for readers seeking a quick yet informative read. ",
"A comprehensive textbook that delves into the physics of stars, galaxies, and cosmology. It's widely used in academic settings and is suitable for readers with a strong interest in the scientific foundations of astrophysics.",
"Hawking explores the origins and structure of the universe, discussing concepts like the Big Bang and black holes. This classic work offers profound insights into cosmology and the nature of time.",
"Sagan takes readers on a journey through space and time, blending scientific facts with philosophical reflections. The book emphasizes the importance of scientific inquiry and our place in the universe.",
"Greene introduces the theory of superstrings and the search for a unified theory of physics. The book is known for its clear explanations of complex topics in theoretical physics.",
"Barbour challenges our conventional understanding of time, proposing that time is an illusion and that the universe is a collection of timeless snapshots. This thought-provoking work delves into the nature of reality and our perception of time.",
"This gripping novel follows astronaut Mark Watney's struggle for survival on Mars after being presumed dead following a storm. Combining humor with scientific accuracy, it showcases human ingenuity in the face of adversity.",
"A classic work that explores the universe's wonders, the history of science, and humanity's place in the cosmos. Sagan's eloquent prose makes complex scientific concepts accessible to all readers.",
"This book delves into the psychology and peculiarities of space travel, offering a humorous and insightful look at the human side of space exploration.",
"Written by a former astronaut, this memoir provides a unique perspective on space missions and the lessons learned that apply to life on Earth",
"A detailed account of the early days of the U.S. space program, focusing on the test pilots and astronauts who embodied the spirit of exploration.",
    ]
    i = -1
    for subdir, _, files in os.walk(folder_path):
    # ترتيب الملفات أبجديًا
        files = sorted(files, key=lambda f: f.lower())  # ترتيب حسب الحروف الصغيرة لتجنب مشاكل الحالة

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # التأكد من نوع الملف
                i+=1
                print(i)
                image_path = os.path.join(subdir, file)
                hog_features = extract_hog_features(image_path).tolist()  # تحويل إلى قائمة لتخزينها
                vgg_features = extract_vgg_features(image_path).tolist()  # تحويل إلى قائمة لتخزينها
                database.append({'image_path': image_path,
                                "titel": titels[i] , 
                                "Author": Author[i],
                                "Description": Description[i],
                                'hog_features': hog_features, 'vgg_features': vgg_features})
    # حفظ قاعدة البيانات في ملف JSON
    with open(save_path, 'w') as json_file:
        json.dump(database, json_file)
    print(f"Database built and saved to {save_path}")
    return database


# === تحميل قاعدة البيانات من ملف JSON === #
def load_image_database(file_path='test.json'):
    with open(file_path, 'r') as json_file:
        database = json.load(json_file)
    # تحويل الميزات المحفوظة إلى NumPy arrays
    for entry in database:
        entry['hog_features'] = np.array(entry['hog_features'])
        entry['vgg_features'] = np.array(entry['vgg_features'])
    print(f"Database loaded from {file_path}")
    return database


# === وظيفة لحساب التشابه بين الصور === #
def calculate_similarity_2(query_features, database_features):
    return cosine_similarity([query_features], [database_features])[0][0]


# === البحث عن الصور المشابهة === #
def search_similar_images(query_image_path, database, top_n=30):
    query_hog_features = extract_hog_features(query_image_path)
    query_vgg_features = extract_vgg_features(query_image_path)

    similarities = []
    for entry in database:
        image_path = entry['image_path']
        hog_features = entry['hog_features']
        vgg_features = entry['vgg_features']

        hog_similarity = calculate_similarity_2(query_hog_features, hog_features)
        vgg_similarity = calculate_similarity_2(query_vgg_features, vgg_features)
        # دمج التشابه من HOG و VGG معاً
        combined_similarity = 0.5 * hog_similarity + 0.5 * vgg_similarity
        similarities.append((image_path, combined_similarity))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:1]

import cv2

def load_and_resize_images_from_paths(results):
    images = []
    for path, similarity in results:
        img = cv2.imread(path)
        if img is not None:
            # img_resized = cv2.resize(img, size)
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
    return images

def show_images_in_figure(images, rows=5, cols=6):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))  # Create a grid of subplots
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')  # Hide axes for a cleaner look
        else:
            ax.axis('off')  # Turn off unused axes
    plt.tight_layout()
    plt.show()

# === Main Execution === #
if __name__ == '__main__':
    # List of relative paths to your images
    main_folder = 'books'
    query_image = 'books/book10image.jpg'

    database_file = 'books.json'
    if os.path.exists(database_file):
        database = load_image_database(database_file)
    else:
        print("Building image database...")
        database = build_image_database(main_folder, save_path=database_file)
    print("Searching for similar images...")
    results = search_similar_images(query_image, database)

    print("Top similar images:")
    images = load_and_resize_images_from_paths(results)

    if images:
        show_images_in_figure(images, rows=5, cols=6)
    else:
        print("No valid images found in the provided paths.")

# === تنفيذ الكود === #
# if __name__ == '__main__':
#     main_folder = 'static/images/form'  # المجلد الذي يحتوي على الصور
#     query_image = 'static/images/form/s.jpg'  # الصورة المراد البحث عنها

#     # === بناء أو تحميل قاعدة البيانات === #
#     database_file = 'database.json'
#     if os.path.exists(database_file):
#         database = load_image_database(database_file)
#     else:
#         print("Building image database...")
#         database = build_image_database(main_folder, save_path=database_file)

#     # === البحث عن الصور المشابهة === #
#     print("Searching for similar images...")
#     results = search_similar_images(query_image, database)

#     # === عرض النتائج === #
#     print("Top similar images:")
#     for image_path, similarity in results:
#         print(f"{image_path} - Similarity: {similarity:.2f}")
    
