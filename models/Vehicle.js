const mongoose = require('mongoose');

const ownerAddressSchema = new mongoose.Schema({
  addressLine1: String,
  addressLine2: String,
  city: String,
  state: String,
  pincode: String
}, { _id: false });

const vehicleSchema = new mongoose.Schema({
  registrationNumber: {
    type: String,
    required: true,
    unique: true,
    uppercase: true,
    trim: true,
    match: [/^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$/, 'Invalid registration number']
  },
  ownercontact: { type: String, required: true, trim: true },
  owneremail: { type: String, trim: true, lowercase: true, match: [/.+@.+\..+/, 'Invalid email address'] },
  make: { type: String, required: true, trim: true },
  model: { type: String, required: true, trim: true },
  variant: { type: String, trim: true },
  transmissions: { type: String, enum: ['Manual', 'Automatic', 'CVT', 'AMT', 'DCT', 'Other'], default: 'Manual' },
  yearOfManufacture: { type: Number, required: true, min: 1900, max: new Date().getFullYear() + 1 },
  fuelType: { type: String, required: true, enum: ['Petrol', 'Diesel', 'CNG', 'Hybrid'] },
  engineCC: { type: Number, required: true, min: 0 },
  seatingCapacity: { type: Number, required: true, min: 1 },
  vehicleClass: { type: String, enum: ['Two-Wheeler', 'Three-Wheeler', 'Four-Wheeler', 'Commercial', 'Heavy Vehicle'], required: true },
  ownerName: { type: String, trim: true },
  ownerAddress: ownerAddressSchema,
  registrationDate: Date,
  chassisNumber: { type: String, unique: true, sparse: true },
  engineNumber: { type: String, unique: true, sparse: true },
  color: { type: String, trim: true },
  status: { type: String, enum: ['Active', 'Expired', 'Scrapped', 'Stolen', 'Transferred'], default: 'Active' },

  fuelConsumptionCombined_kmpl: { type: Number, min: 0 },

  refuelLitresHistory: [{ litres: Number, date: Date }],
  pricePaidHistory: [{ amount: Number, date: Date }],
  odometerHistory: [{ odometer: Number, date: Date }],
  visitCount: { type: Number, default: 0 },

}, {
  timestamps: true
});

vehicleSchema.index({ registrationNumber: 1 });

module.exports = mongoose.model('Vehicle', vehicleSchema);
