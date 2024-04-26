# CS 394: Agile Software Development 2024 Winter

Instructor: Christopher Riesbeck

Please follow both [Northwestern University's Principles of Academic Integrity](https://www.northwestern.edu/provost/policies-procedures/academic-integrity/principles.html#:~:text=Academic%20integrity%20at%20Northwestern%20is,integrity%20is%20a%20fundamental%20commitment.) and the [Northwestern Department of Computer Science's Academic Integrity Policy](https://catalogs.northwestern.edu/sps/graduate-academic-policies-procedures/academic-integrity/)

## Past Assignments

[MeetTogether](https://github.com/394-w24/MeetTogether)

[NoteDoctor](https://github.com/394-w24/NoteDoctor)

### About The Project
github repo name: 394-w24 / NoteDoctor

The project is created by team Red in CS394 course taught by Professor Riesbeck in winter 2024, in collaboration with Northwestern Master of Product Design and Development Management MPD program.

- **Innovative Web Application**: "NoteDoctor" is designed to revolutionize the healthcare experience.
- **Privacy Priority**: Prioritizes patient privacy throughout the consultation process.
- **Secure Platform**: Offers a secure platform for displaying patients' medical conditions and history.
- **User-Friendly Interface**: Ensures a user-friendly experience, making consultations more convenient and protecting sensitive information.
- **Efficiency in Clinic Operations**: Enhances clinic operations by monitoring the availability of consultation rooms, which helps in organizing patient flow and reducing wait times.
- **Modern Technology**: Developed with cutting-edge technologies, including Vite and React for the frontend, and Firebase for the backend.
- **Commitment to Healthcare**: Committed to delivering a seamless, efficient, and secure healthcare service for both patients and healthcare providers.

<a name="readme-top"></a>

<details open>
<summary>Table of Contents</summary>
<ul>
  <li><a href="#about-the-project">About The Project</a></li>
  <li><a href="#built-with">Built With</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#prerequisites">Prerequisites</a></li>
  <li><a href="#installation-and-running-the-app">Installation and Running the App</a></li>
  <li><a href="#detailed-firebase-setup">Detailed Firebase Setup</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#roadmap">Roadmap</a></li>
  <li><a href="#contributing">Contributing</a></li>
  <li><a href="#contact">Contact</a></li>
  <li><a href="#acknowledgements">Acknowledgements</a></li>
</ul>
</details>


### Built With

* [![React][React.js]][React-url]
* [![Vite][Vite.js]][Vite-url]
* [![Firebase][Firebase.js]][Firebase-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
  
## Features:

- Dynamic room assignment system
- Patient privacy verification
- Customized welcome page for patients
- Symptom add-on functionality

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites
```
node version >= 16
```

### Installation and Running the App
1. **Clone the repository:**
   ```bash
   git clone https://github.com/394-w24/NoteDoctor.git
   cd note-doctor
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the application:**
   ```bash
   npm start
   ```
   Please make sure you set up Firebase and then start to run. This command runs the app in development mode. Visit [http://localhost:5173](http://localhost:5173) in your browser. The app will automatically reload if you change any of the source files.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Detailed Firebase Setup

1. **Create a Firebase Account and Project:**
   - Visit [Firebase](https://firebase.google.com/) and sign up or log in.
   - Click on "Go to console" at the top right corner.
   - Click on "Add project" and follow the steps to create a new Firebase project.

2. **Get Your Firebase Configuration:**
   - In the Firebase console, select your project.
   - Click on "Project settings" in the left menu.
   - Find your Firebase project's configuration in the "General" tab under the "Your apps" section by adding a new web app if necessary.
   - Click on the "</>" icon to register a new web app and follow the prompts.
   - After the app is registered, you will see your Firebase configuration keys which look like this:
     ```javascript
     const firebaseConfig = {
       apiKey: "YOUR_API_KEY",
       authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
       projectId: "YOUR_PROJECT_ID",
       storageBucket: "YOUR_PROJECT_ID.appspot.com",
       messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
       appId: "YOUR_APP_ID",
       measurementId: "YOUR_MEASUREMENT_ID"
     };
     ```

3. **Configure Firebase in Your Application:**
   - Create a file named `firebase.js` in your project's source directory (e.g., `/src`).
   - Paste the Firebase configuration code snippet you obtained from the Firebase console into `firebase.js`.
   - Make sure to replace the placeholder values in the configuration with your actual Firebase project details.

4. **Install Firebase SDK:**
   - Run the following command in your project directory to install the Firebase package:
     ```bash
     npm install firebase
     ```
   - In `firebase.js`, initialize Firebase using the config object:
     ```javascript
     import { initializeApp } from 'firebase/app';
     // Your firebaseConfig from step 2
     const app = initializeApp(firebaseConfig);
     ```

5. **Import Starting Data into Firestore:**
   - Ensure you have a `data.json` file with the data you want to import into Firestore.
   - Install the `node-firestore-import-export` tool:
     ```bash
     npm install -g node-firestore-import-export
     ```
   - Generate a new private key for your Firebase service account in the Firebase console under "Project settings" > "Service accounts" and download it.
   - Import your data into Firestore using the command line:
     ```bash
     firestore-import -a path/to/your/credentials.json -b path/to/your/data.json
     ```
     Replace `path/to/your/credentials.json` and `path/to/your/data.json` with the actual paths to your files.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Note Doctor is engineered to enhance the user experience across various roles within the healthcare ecosystem. Each screen is purpose-built, focusing on ease of use, security, and efficiency.

### Screen 1: Patient Display View & Modal

![1](https://github.com/394-w24/NoteDoctor/assets/85666623/5f14ef7f-0dec-4f7e-bb0d-a8267f17f222)
![5](https://github.com/394-w24/NoteDoctor/assets/85666623/4bc9e1aa-ae20-4f48-a5d2-cce098b28ab4)


- **Objective**: This interactive screen is tailored for patients to provide them with a detailed overview of their scheduled appointments and information about their care team.
- **Security Measure**: To protect patient privacy, a birthdate verification is required as a security measure before any medical information is displayed.
- **Issue Reporting**: Patients can conveniently report additional issues either by entering text or using the intuitive "click and add" functionality.

### Screen 2: Nurse Check-In View
![2](https://github.com/394-w24/NoteDoctor/assets/85666623/9277b0fc-a716-48d7-be1f-96b3e6e2b532)

- **Objective**: Nurses use this screen to streamline the check-in process by assigning patients to rooms, helping manage patient intake and the overall flow within the facility.

### Screen 3: Room Overview
![3](https://github.com/394-w24/NoteDoctor/assets/85666623/123d24ee-96a3-42b7-848a-f50415a680d5)


- **Objective**: This screen offers a real-time overview of room statuses, providing valuable information on room availability, which assists in resource management and allocation.

### Screen 4: Backdoor View
![4](https://github.com/394-w24/NoteDoctor/assets/85666623/2ed8718e-412a-43cf-9503-543dfad8a947)

- **Objective**: This screen is added for demonstration purposes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Week 1 (2/9-2/15)
  - [x] Meet the developers
  - [x] Overview of Concept
  - [x] Requirements Refinement
  - [ ] Product team to create UI mock to enable engineering team for build phase


- [x] Week 2 (2/16-2/22)
  - [x] Product team to create UI mock to enable engineering team for build phase
  - [ ] Engineering team to create MVP version of app- (2x) meeting with product team


- [X] Week 3 (2/23-2/29)
  - [x] Engineering team to create MVP version of app- (2x) meeting with product team
    
- [X] Week 4 (3/1-3/7)
  - [x] Go/no go decision with product team

- [ ] Future Work and known issues
  - [ ] Multi-screen responsiveness for tablet and wide screen use
  - [ ] Fix issue where nurse can assign patient to already occupied room

<p align="right">(<a href="#readme-top">back to top</a>)</p>
        
## Contributing

Contributions to NoteDoctor are greatly appreciated! 

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". 

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/YourFeature)
3. Commit your Changes (git commit -m 'Add some YourFeature')
4. Push to the Branch (git push origin feature/YourFeature)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Professor Christopher Riesbeck: c-riesbeck@northwestern.edu <br/>
Developer Contact: Aldierygonzalez2024@u.northwestern.edu

CS394 Team Roster:
- Perry Benyella - Aldiery Rene Gonzalez - Zhuoyuan Li - Rodney David Reichert - Dong Shu - Quanyue Xie - Haoyang Yuan - Kelly Mei

MPD Team Roster:

- Emily Zarefsky - Dani Salonen - Sarah Bennett

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgements

Christopher K Riesbeck

W24 CS 394 Team Red

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vite.js]: https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E
[Vite-url]: https://vitejs.dev/
[Firebase.js]: https://img.shields.io/badge/Firebase-ffca28?style=for-the-badge&logo=firebase&logoColor=black
[Firebase-url]: https://firebase.google.com/
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
