// seed.js
const mongoose = require('mongoose');
const { Vehicle, PetrolPumpVisit, VehicleMetrics, PUCDetails } = require('./models');

// 1. Connect to MongoDB
mongoose.connect('mongodb://127.0.0.1:27017/vehicleDB')
  .then(() => console.log("MongoDB connected"))
  .catch(err => console.error(err));

// 2. Create seed data
async function seedData() {
  try {
    // Clear existing data
    await Vehicle.deleteMany({});
    await PetrolPumpVisit.deleteMany({});
    await VehicleMetrics.deleteMany({});
    await PUCDetails.deleteMany({});

    // ---------------- Vehicles ----------------
    const vehicles = await Vehicle.insertMany([
      {
        registrationNumber: "RJ14AB1234", ownercontact: "9876543210", make: "Maruti", model: "Swift",
        variant: "VXI", transmissions: "Manual", yearOfManufacture: 2020, fuelType: "Petrol", engineCC: 1197,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Rahul Sharma",
        ownerAddress: { addressLine1: "123", city: "Jaipur", state: "Rajasthan", pincode: "302001" }, color: "Red"
      },
      {
        registrationNumber: "DL55AX4321", ownercontact: "9998887776", make: "Honda", model: "City",
        variant: "ZX", transmissions: "Automatic", yearOfManufacture: 2019, fuelType: "Diesel", engineCC: 1498,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Ankit Gupta", color: "White"
      },
      {
        registrationNumber: "MH12AB3456", ownercontact: "8887776665", make: "Hyundai", model: "i20",
        variant: "Sportz", transmissions: "Manual", yearOfManufacture: 2021, fuelType: "Petrol", engineCC: 1199,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Neha Kulkarni", color: "Blue"
      },
      {
        registrationNumber: "GJ01CD7890", ownercontact: "7776665554", make: "Tata", model: "Nexon",
        transmissions: "AMT", yearOfManufacture: 2022, fuelType: "Diesel", engineCC: 1497,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Ravi Patel", color: "Grey"
      },
      {
        registrationNumber: "KA03EF1111", ownercontact: "9991112223", make: "Kia", model: "Seltos",
        transmissions: "DCT", yearOfManufacture: 2023, fuelType: "Petrol", engineCC: 1353,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Priya Iyer", color: "Black"
      },
      {
        registrationNumber: "TN10GH2222", ownercontact: "7773331112", make: "Toyota", model: "Innova",
        transmissions: "Manual", yearOfManufacture: 2018, fuelType: "Diesel", engineCC: 2393,
        seatingCapacity: 7, vehicleClass: "Four-Wheeler", ownerName: "Suresh Kumar", color: "Silver"
      },
      {
        registrationNumber: "UP32JK3333", ownercontact: "9123456780", make: "Mahindra", model: "XUV700",
        transmissions: "Automatic", yearOfManufacture: 2022, fuelType: "Petrol", engineCC: 1999,
        seatingCapacity: 7, vehicleClass: "Four-Wheeler", ownerName: "Amit Verma", color: "White"
      },
      {
        registrationNumber: "WB20LM4444", ownercontact: "8989898989", make: "Suzuki", model: "Baleno",
        transmissions: "Manual", yearOfManufacture: 2021, fuelType: "Petrol", engineCC: 1197,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Rohit Das", color: "Red"
      },
      {
        registrationNumber: "HR26NP5555", ownercontact: "9765432100", make: "Hyundai", model: "Creta",
        transmissions: "Automatic", yearOfManufacture: 2020, fuelType: "Diesel", engineCC: 1493,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Karan Singh", color: "Blue"
      },
      {
        registrationNumber: "PB08QR6666", ownercontact: "9988776655", make: "Ford", model: "Ecosport",
        transmissions: "Manual", yearOfManufacture: 2017, fuelType: "Petrol", engineCC: 1499,
        seatingCapacity: 5, vehicleClass: "Four-Wheeler", ownerName: "Simran Kaur", color: "Orange"
      }
    ]);

    // ---------------- Petrol Pump Visits ----------------
    const pumpVisits = [];
    vehicles.forEach((v, i) => {
      pumpVisits.push({
        registrationNumber: v.registrationNumber,
        visits: [
          {
            visitDate: new Date(),
            petrolPumpName: "Indian Oil",
            petrolPumpLocation: { address: "Station Road", city: "City" + (i+1), state: "State" + (i+1) },
            fuelType: v.fuelType,
            litresFilled: 30 + i,
            pricePaid: 3000 + i * 50,
            pricePerLitre: 100,
            paymentMethod: "UPI",
            odometerReading: 10000 + i * 500
          }
        ]
      });
    });
    await PetrolPumpVisit.insertMany(pumpVisits);

    // ---------------- Vehicle Metrics ----------------
    const metrics = [];
    vehicles.forEach((v, i) => {
      metrics.push({
        registrationNumber: v.registrationNumber,
        forMonth: "2025-09",
        totalFuelThisMonth: 50 + i * 5,
        refuelCountThisMonth: 2,
        avgRefuelLitresThisMonth: 25 + i,
        totalDistanceThisMonth: 1000 + i * 100,
        avgDailyDistanceThisMonth: 35 + i,
        fuelConsumptionConsistency: 2 + i * 0.2,
        lastOdometerReading: 12000 + i * 400,
        visitCount: 2
      });
    });
    await VehicleMetrics.insertMany(metrics);

    // ---------------- PUC Details ----------------
    const puc = [];
    vehicles.forEach((v, i) => {
      puc.push({
        registrationNumber: v.registrationNumber,
        certificateNumber: "PUC" + (1000 + i),
        issueDate: new Date("2025-08-01"),
        validUntil: new Date("2026-02-01"),
        emissionReadings: { co: 0.2 + i*0.01, hc: 18 + i, co2: 14 + i, opacity: 1.0 + i*0.1 },
        testingCenter: { name: "Center " + (i+1), address: "Area " + (i+1), licenseNumber: "LIC" + (i+1) },
        status: "Active"
      });
    });
    await PUCDetails.insertMany(puc);

    console.log("✅ 10 records per table inserted successfully!");
    mongoose.connection.close();
  } catch (err) {
    console.error(err);
    mongoose.connection.close();
  }
}

seedData();
