export interface SupplierData {
  supplier: string;
  sales: number;
  share: number;
}

export interface ValidationConfig {
  topSuppliers: SupplierData[];
  extendedSuppliers?: string[];
  allowUnknownSuppliers?: boolean;
}

export interface ValidationResult {
  supplier_found: boolean;
  lost_supplier: string;
  lost_share: number;
  lost_capacity?: number;
  lost_capacity_units?: number;
  warning?: string;
  error?: string;
}

function _normalize(name?: string): string {
  return (name || "").trim().toUpperCase();
}

export function validateSupplier(disrupted: string, config: ValidationConfig): ValidationResult {
  const top = config.topSuppliers || [];
  const disruptedN = _normalize(disrupted);

  // default result
  const base: ValidationResult = {
    supplier_found: false,
    lost_supplier: disrupted,
    lost_share: 0,
  };

  if (!disrupted || !disrupted.trim()) {
    base.error = "No disrupted supplier provided";
    return base;
  }

  // check top suppliers
  const topMatch = top.find(s => _normalize(s.supplier) === disruptedN);
  if (topMatch) {
    const share = Number(topMatch.share || 0);
    return {
      supplier_found: true,
      lost_supplier: topMatch.supplier,
      lost_share: share,
      lost_capacity: Number(topMatch.sales || 0),
      lost_capacity_units: 0, // leave unit conversion to caller if needed
    };
  }

  // check extended suppliers
  if (config.extendedSuppliers && config.extendedSuppliers.map(_normalize).includes(disruptedN)) {
    return {
      supplier_found: true,
      lost_supplier: disrupted,
      lost_share: 0,
      warning: "Found in extendedSuppliers; share/capacity unknown",
    };
  }

  // allow unknown suppliers mode
  if (config.allowUnknownSuppliers) {
    return {
      supplier_found: true,
      lost_supplier: disrupted,
      lost_share: 0,
      warning: "Unknown supplier accepted due to allowUnknownSuppliers=true",
    };
  }

  // strict mode: not found
  return {
    supplier_found: false,
    lost_supplier: disrupted,
    lost_share: 0,
    error: "Supplier not found in topSuppliers or extendedSuppliers",
  };
}
