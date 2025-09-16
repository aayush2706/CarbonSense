const mongoose = require('mongoose');

const pucDetailsSchema = new mongoose.Schema({
  registrationNumber: { type: String, required: true, uppercase: true, trim: true, index: true },
  certificateNumber: { type: String, required: true, unique: true },
  issueDate: { type: Date, required: true },
  validUntil: { type: Date, required: true },
  emissionReadings: {
    co: { type: Number, min: 0 },
    hc: { type: Number, min: 0 },
    co2: { type: Number, min: 0 },
    opacity: { type: Number, min: 0 }
  },
  testingCenter: {
    name: String,
    address: String,
    licenseNumber: String
  },
  status: { type: String, enum: ['Active', 'Expired', 'Revoked'], default: 'Active' },
}, {
  timestamps: true
});

pucDetailsSchema.index({ registrationNumber: 1 });
pucDetailsSchema.index({ validUntil: 1 });

module.exports = mongoose.model('PUCDetails', pucDetailsSchema);
