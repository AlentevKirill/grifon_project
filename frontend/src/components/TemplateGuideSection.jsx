import { getPredictionTemplateUrl } from "../api/client";

const templateColumns = [
  {
    name: "Customer_ID",
    expectedFormat: "String or integer identifier",
    description: "A particular identifier for every customer within the bank's system.",
  },
  {
    name: "Customer_Name",
    expectedFormat: "Text (full name)",
    description: "The name of the consumer making the transaction.",
  },
  {
    name: "Gender",
    expectedFormat: "Categorical text (Male, Female, Other)",
    description: "The gender of the consumer (e.g., Male, Female, Other).",
  },
  {
    name: "Age",
    expectedFormat: "Integer (years)",
    description: "The age of the consumer at the time of the transaction.",
  },
  {
    name: "State",
    expectedFormat: "Text",
    description: "The nation in which the patron resides.",
  },
  {
    name: "City",
    expectedFormat: "Text",
    description: "The metropolis wherein the client is living.",
  },
  {
    name: "Bank_Branch",
    expectedFormat: "Text",
    description: "The specific financial institution branch wherein the consumer holds their account.",
  },
  {
    name: "Account_Type",
    expectedFormat: "Categorical text (Savings, Checking, etc.)",
    description: "The kind of account held with the aid of the customer (e.g., Savings, Checking).",
  },
  {
    name: "Transaction_ID",
    expectedFormat: "String or integer identifier",
    description: "A particular identifier for each transaction.",
  },
  {
    name: "Transaction_Date",
    expectedFormat: "Date (YYYY-MM-DD preferred)",
    description: "The date on which the transaction passed off.",
  },
  {
    name: "Transaction_Time",
    expectedFormat: "Time (HH:MM:SS preferred)",
    description: "The specific time the transaction became initiated.",
  },
  {
    name: "Transaction_Amount",
    expectedFormat: "Decimal number",
    description: "The financial value of the transaction.",
  },
  {
    name: "Merchant_ID",
    expectedFormat: "String or integer identifier",
    description: "A particular identifier for the merchant worried within the transaction.",
  },
  {
    name: "Transaction_Type",
    expectedFormat: "Categorical text (Withdrawal, Deposit, Transfer, etc.)",
    description: "The nature of the transaction (e.g., Withdrawal, Deposit, Transfer).",
  },
  {
    name: "Merchant_Category",
    expectedFormat: "Categorical text (Retail, Online, Travel, etc.)",
    description: "The class of the merchant (e.g., Retail, Online, Travel).",
  },
  {
    name: "Account_Balance",
    expectedFormat: "Decimal number",
    description: "The balance of the customer's account after the transaction.",
  },
  {
    name: "Transaction_Device",
    expectedFormat: "Categorical text (Mobile, Desktop, etc.)",
    description: "The tool utilized by the consumer to perform the transaction (e.g., Mobile, Desktop).",
  },
  {
    name: "Transaction_Location",
    expectedFormat: "Text (address or latitude,longitude)",
    description: "The geographical vicinity (e.g., latitude, longitude) of the transaction.",
  },
  {
    name: "Device_Type",
    expectedFormat: "Categorical text (Smartphone, Laptop, etc.)",
    description: "The kind of device used for the transaction (e.g., Smartphone, Laptop).",
  },
  {
    name: "Is_Fraud",
    expectedFormat: "Binary integer (0 or 1)",
    description: "A binary indicator (1 or 0) indicating whether or not the transaction is fraudulent or not.",
  },
  {
    name: "Transaction_Currency",
    expectedFormat: "Currency code text (USD, EUR, etc.)",
    description: "The currency used for the transaction (e.g., USD, EUR).",
  },
  {
    name: "Customer_Contact",
    expectedFormat: "Text or phone-number string",
    description: "The contact variety of the client.",
  },
  {
    name: "Transaction_Description",
    expectedFormat: "Short text",
    description: "A brief description of the transaction (e.g., buy, switch).",
  },
  {
    name: "Customer_Email",
    expectedFormat: "Email address",
    description: "The e-mail cope with related to the consumer's account.",
  },
];

function TemplateGuideSection() {
  return (
    <section className="card template-card">
      <h2>Excel template and data format guide</h2>
      <p className="hint">
        Download the template and fill it with transaction records before upload. The file must be in XLSX format.
      </p>

      <div className="template-download-row">
        <a className="template-link" href={getPredictionTemplateUrl()} download>
          Download Excel template (.xlsx)
        </a>
      </div>

      <div className="template-table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Column</th>
              <th>Expected format</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {templateColumns.map((column) => (
              <tr key={column.name}>
                <td>{column.name}</td>
                <td>{column.expectedFormat}</td>
                <td>{column.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default TemplateGuideSection;
