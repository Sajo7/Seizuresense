require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const port = 3000;

// Initialize Google Generative AI
const genAI = new GoogleGenerativeAI("AIzaSyBZKAHN1dtuYL0qwGltL8RMy-wsKZ9v5Bc");

// MongoDB connection
console.log('MONGODB_URI:', process.env.MONGODB_URI);
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('Error connecting to MongoDB:', err));

// Define schema and model for patient data
const patientSchema = new mongoose.Schema({
  name: String,
  email: String,
  age: Number,
  seizureHistory: String,
  medications: String,
  symptoms: String
});
const Patient = mongoose.model('Patient', patientSchema);

// Middleware setup
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// File upload configuration
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, file.originalname)
});
const upload = multer({ storage: storage });

// Function to clear uploads folder
function clearUploadsFolder(directory) {
  fs.readdir(directory, (err, files) => {
    if (err) return console.error('Error reading directory:', err);
    files.forEach(file => {
      const filePath = path.join(directory, file);
      fs.stat(filePath, (err, stats) => {
        if (err) return console.error('Error getting file stats:', err);
        if (stats.isFile()) {
          fs.unlink(filePath, err => {
            if (err) console.error('Error deleting file:', err);
          });
        }
      });
    });
  });
}

// File upload and prediction route
app.post('/upload', async (req, res) => {
    clearUploadsFolder(uploadDir);
  
    upload.array('files')(req, res, async function (err) {
      if (err) return res.status(500).send('Error uploading files: ' + err);
  
      exec('python prediction_code.py', async (error, stdout, stderr) => {
        if (error || stderr) {
          console.error('Error in prediction script:', error || stderr);
          return res.status(500).send('Error during prediction.');
        }
  
        // Clean the output to extract just the prediction result
        const predictionResult = stdout.trim().split('\n').pop();  // Extract only the last line of the output
  
        if (predictionResult === "You are diagnosed with epileptic seizure.") {
          try {
            // Use Generative AI to fetch preventive measures for epilepsy
            const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
            const prompt = "Provide preventive measures and dos and don'ts for someone diagnosed with epileptic seizure.";
            const result = await model.generateContent(prompt);
  
            const responseText = result.response.text(); // This should extract the text from the response object
            res.json({
              predictionResult,
              advice: responseText // Include the preventive advice from Generative AI
            });
          } catch (err) {
            console.error('Error generating content with Generative AI:', err);
            res.status(500).send('Error fetching preventive measures.');
          }
        } else {
          res.json({
            predictionResult,
            advice: "No preventive measures required." // For other predictions, no specific advice
          });
        }
      });
    });
  });
  

// Patient data submission route
app.post('/submit', (req, res) => {
  const { name, email, age, seizureHistory, medications, symptoms } = req.body;

  const newPatient = new Patient({
    name,
    email,
    age,
    seizureHistory,
    medications,
    symptoms
  });

  newPatient.save()
    .then(() => {
      console.log('Patient data saved successfully!');
      res.redirect('/page2.html');
    })
    .catch(err => {
      console.error('Error saving patient data:', err);
      res.status(500).send('Server error');
    });
});

// Start server
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'html.html'));
});
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
